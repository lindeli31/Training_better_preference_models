"""
opro_position_bias.py
---------------------
OPRO (Optimization by PROmpting) for reducing position bias in LLM judges.

The idea: use the LLM itself to iteratively generate better system prompts.
Each candidate prompt is scored, aggregated over a fixed eval subset, as:
    score = w_c * consistency + w_a * accuracy - length_hinge(|prompt|)
with w_c=0.5, w_a=0.5, L_target=2200, λ_L=1e-4, saturation at 3000 chars.
The meta-LM sees the history of (prompt, score, full metrics) triples and
tries to propose a prompt that scores higher.

Gold score for each response (defined in src/dataset.py::_score):
    score(row) = (helpfulness + correctness + coherence) / 3.0
gold_label is derived from the two responses' scores:
    A if score_a > score_b,  B if score_b > score_a,  C if equal.

Final reporting uses the SAME compute_position_bias used by GEPA plus an
accuracy column computed over both AB and BA orders (BA's expected gold is
the flipped gold label). This makes OPRO numbers directly comparable to the
GEPA numbers stored in results/gepa/gepa_results.json.

Pipeline:
  1. Baseline = SYSTEM_EXPERT_RATER (same seed as GEPA), evaluated on
     train + val with the full metric.
  2. Initialise OPRO history with 2 seed prompts.
  3. For N iterations: show history to LLM → generate candidate →
     evaluate consistency on a small eval_subset (fast proxy).
  4. Pick the best-scoring prompt on the subset.
  5. Re-evaluate the winner on full train + val with the full metric.
  6. Save baseline_train, baseline_val, train, val, history, best_prompt.
"""

# Standard library imports
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from src.datasets.dataset import PairRecord
from src.core.inference_client import SwissAIClient
from src.eval.metrics import compute_position_bias
from src.core.templates import _build_user_prompt
from src.runall_prompt_optimization.scoring import compute_score, _FLIP, SEED_PROMPTS
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluate position consistency of a system prompt
# ---------------------------------------------------------------------------

async def evaluate_full_metrics(
    client: SwissAIClient,
    pairs: list[PairRecord],
    system_prompt: str,
    criterion: str = "better overall, considering helpfulness, correctness, and coherence",
    experiment_id: str = "opro_full",
) -> dict:
    """Evaluate a system prompt on pairs and return the SAME metric dict as
    GEPA's evaluate_judge: position_consistency, position_bias_rate,
    bias_toward_first_position (from compute_position_bias), plus
    accuracy_ab, accuracy_ba, accuracy (averaged over both orders, BA
    compared against the flipped gold).

    Used for baseline + final evaluation so OPRO and GEPA numbers line up.
    """
    calls = []
    for pair in pairs:
        user_ab = _build_user_prompt(pair.prompt, pair.response_a, pair.response_b, criterion)
        calls.append(dict(
            system_prompt=system_prompt, user_prompt=user_ab,
            prompt_id=pair.prompt_id, experiment_id=experiment_id, condition="AB",
        ))
        flipped = pair.flipped()
        user_ba = _build_user_prompt(flipped.prompt, flipped.response_a, flipped.response_b, criterion)
        calls.append(dict(
            system_prompt=system_prompt, user_prompt=user_ba,
            prompt_id=pair.prompt_id, experiment_id=experiment_id, condition="BA",
        ))

    results = await client.batch_judge(calls)

    # Build the JSONL-style records compute_position_bias expects, and
    # track per-order accuracy against gold (BA uses flipped gold).
    records = []
    gold_by_pid = {p.prompt_id: p.gold_label for p in pairs}
    ab_correct = ab_total = 0
    ba_correct = ba_total = 0
    for r in results:
        records.append({"prompt_id": r.prompt_id, "condition": r.condition, "label": r.label})
        if r.label is None:
            continue
        gold = gold_by_pid[r.prompt_id]
        if r.condition == "AB":
            ab_total += 1
            if r.label == gold:
                ab_correct += 1
        elif r.condition == "BA":
            ba_total += 1
            if r.label == _FLIP[gold]:
                ba_correct += 1

    metrics = compute_position_bias(records)
    metrics["accuracy_ab"] = round(ab_correct / ab_total, 4) if ab_total else 0.0
    metrics["accuracy_ba"] = round(ba_correct / ba_total, 4) if ba_total else 0.0
    total_correct = ab_correct + ba_correct
    total_calls   = ab_total   + ba_total
    metrics["accuracy"] = round(total_correct / total_calls, 4) if total_calls else 0.0
    return metrics


async def evaluate_position_bias(
    client: SwissAIClient,
    pairs: list[PairRecord],
    system_prompt: str,
    criterion: str = "better overall, considering helpfulness, correctness, and coherence",
) -> float:
    """
    Test how consistent a system prompt is when response order is flipped.

    For each pair, sends two judge calls:
      - AB order: response_a first, response_b second
      - BA order: response_b first, response_a second

    A consistent judge should prefer the same underlying response regardless
    of order. E.g. if AB → "A", then BA should → "B" (same response, new position).

    Returns a float between 0.0 (always flips) and 1.0 (perfectly consistent).
    """

    # Build one AB call and one BA call per pair
    calls = []
    for pair in pairs:
        # Original order: A first, B second
        user_p = _build_user_prompt(pair.prompt, pair.response_a, pair.response_b, criterion)
        calls.append(dict(
            system_prompt=system_prompt, user_prompt=user_p,
            prompt_id=pair.prompt_id, experiment_id="opro_eval",
            condition="AB",
        ))

        # Flipped order: B first, A second
        flipped = pair.flipped()
        user_p2 = _build_user_prompt(flipped.prompt, flipped.response_a, flipped.response_b, criterion)
        calls.append(dict(
            system_prompt=system_prompt, user_prompt=user_p2,
            prompt_id=pair.prompt_id, experiment_id="opro_eval",
            condition="BA",
        ))

    # Send all calls to the API in parallel
    results = await client.batch_judge(calls)

    # Collect labels: one per prompt_id per condition
    ab_labels = {}
    ba_labels = {}
    for r in results:
        if r.condition == "AB":
            ab_labels[r.prompt_id] = r.label
        else:
            ba_labels[r.prompt_id] = r.label

    # Compare AB vs BA labels for each pair
    # flip_map: if AB says "A" (first is better), BA should say "B" (same response, now second)
    flip_map = {"A": "B", "B": "A", "C": "C"}
    consistent = 0
    total = 0
    for pid in ab_labels:
        if pid not in ba_labels:
            continue
        a_label = ab_labels[pid]
        b_label = ba_labels[pid]
        if a_label is None or b_label is None:
            continue
        total += 1
        if a_label == flip_map.get(b_label):
            consistent += 1

    return consistent / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Meta-prompt: ask the LLM to generate a better system prompt
# ---------------------------------------------------------------------------

# This is the system prompt for the "prompt engineer" LLM call.
# It tells the LLM: look at previous prompts and their scores, write a better one.
META_SYSTEM = (
    "You are a prompt engineer. Your task is to write a system prompt for an "
    "LLM judge that compares two responses (A and B) to a user query.\n\n"
    "The judge must output A, B, or C (tie). The goal is to MINIMISE POSITION BIAS: "
    "the judge should give the same verdict regardless of which response is shown first.\n\n"
    "Below are previous system prompts and their position consistency scores "
    "(higher = less biased, 1.0 = perfect). Generate a NEW system prompt that "
    "scores higher. Focus on instructions that reduce order effects.\n\n"
    "Output ONLY the new system prompt text, nothing else."
)


def _build_meta_prompt(history: list[dict]) -> str:
    """
    Build the user message for the meta-optimizer.
    Shows all previous attempts with their scores so the LLM can learn
    what works and what doesn't. When an entry has a full metrics dict
    (new format), we surface all three dimensions — consistency (what OPRO
    optimises), accuracy, and bias_rate — so the meta-LM can reason about
    the consistency/accuracy trade-off rather than only chasing the scalar.
    """
    lines = ["Previous attempts (prompt → metrics). Higher consistency is "
             "better; higher accuracy is better; lower bias_rate is better.\n"]
    for i, entry in enumerate(history, 1):
        m = entry.get("metrics")
        if m:
            header = (f"--- Attempt {i} (score={entry['score']:.3f} | "
                      f"consistency={m['position_consistency']:.3f}, "
                      f"accuracy={m['accuracy']:.3f}, "
                      f"bias_rate={m['position_bias_rate']:.3f}, "
                      f"len={len(entry['prompt'])}) ---")
        else:
            header = f"--- Attempt {i} (score: {entry['score']:.3f}) ---"
        lines.append(header)
        lines.append(entry["prompt"])
        lines.append("")
    lines.append("Generate a better system prompt:")
    return "\n".join(lines)


async def generate_candidate(
    client: SwissAIClient,
    history: list[dict],
) -> str:
    """
    Ask the LLM to propose a new system prompt based on the history
    of previous attempts and their scores.
    Returns the raw generated text as the new candidate prompt.
    """
    user_prompt = _build_meta_prompt(history)
    response = await client.judge(
        system_prompt=META_SYSTEM,
        user_prompt=user_prompt,
        prompt_id="meta",
        experiment_id="opro_meta",
        condition="generate",
    )
    return response.raw_text.strip()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OproResult:
    best_prompt: str          # the system prompt with highest subset score
    best_train_score: float   # its GEPA-style score on the eval_subset (fast proxy)
    baseline_train: dict      # full metrics of SYSTEM_EXPERT_RATER on train pairs
    baseline_val: dict        # full metrics of SYSTEM_EXPERT_RATER on val pairs
    train: dict               # full metrics of best_prompt on train pairs
    val: dict                 # full metrics of best_prompt on val pairs
    history: list[dict]       # [{"prompt": ..., "score": ...}, ...]


# ---------------------------------------------------------------------------
# Main OPRO loop
# ---------------------------------------------------------------------------

async def run_opro(
    client: SwissAIClient,
    train_pairs: list[PairRecord],
    val_pairs: list[PairRecord],
    n_iterations: int = 10,
    eval_pairs: int = 30,
    seed: int = 42,
    output_dir: Path = Path("results/opro"),
    proposer_client: SwissAIClient | None = None,
    precomputed_baseline_train: dict | None = None,
    precomputed_baseline_val: dict | None = None,
) -> OproResult:
    """
    Run the full OPRO optimisation loop.

    1. Sample a fixed subset of train pairs for fast evaluation
    2. Evaluate 2 seed prompts as starting points
    3. For each iteration: generate a new candidate, evaluate it, add to history
    4. Pick the prompt with the highest train score
    5. Validate it on the full validation set
    6. Save results to JSON

    Args:
        train_pairs: all training pairs (a subset is sampled for evaluation)
        val_pairs:   held-out pairs for final validation
        n_iterations: number of candidate prompts to generate
        eval_pairs:  size of the train subset used per evaluation
        seed:        random seed for reproducibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample a fixed subset of train pairs — used for ALL iterative
    # evaluations (seed + candidates) so that scores are comparable.
    rng = random.Random(seed)
    eval_subset = rng.sample(train_pairs, min(eval_pairs, len(train_pairs)))

    if precomputed_baseline_train is not None and precomputed_baseline_val is not None:
        baseline_train = precomputed_baseline_train
        baseline_val   = precomputed_baseline_val
        logger.info(
            "Using pre-computed shared baseline: "
            "train cons=%.3f acc=%.3f | val cons=%.3f acc=%.3f",
            baseline_train["position_consistency"], baseline_train["accuracy"],
            baseline_val["position_consistency"],   baseline_val["accuracy"],
        )
    else:
        baseline_prompt = SEED_PROMPTS[0]
        logger.info("Evaluating OPRO baseline (SEED_PROMPTS[0]) on %d train pairs", len(train_pairs))
        baseline_train = await evaluate_full_metrics(client, train_pairs, baseline_prompt, experiment_id="opro_baseline_train")
        logger.info(
            "Baseline (train) consistency=%.3f accuracy=%.3f bias_rate=%.3f",
            baseline_train["position_consistency"], baseline_train["accuracy"],
            baseline_train["position_bias_rate"],
        )
        logger.info("Evaluating OPRO baseline on %d val pairs", len(val_pairs))
        baseline_val = await evaluate_full_metrics(client, val_pairs, baseline_prompt, experiment_id="opro_baseline_val")
        logger.info(
            "Baseline (val)   consistency=%.3f accuracy=%.3f bias_rate=%.3f",
            baseline_val["position_consistency"], baseline_val["accuracy"],
            baseline_val["position_bias_rate"],
        )

    # Two seed prompts that SEED the OPRO history (start of the search).
    # Same seeds as GEPA so the two methods start from comparable points.
    seed_prompts = SEED_PROMPTS

    # Evaluate seed prompts on the FAST subset to initialise the history.
    # Score stored in history is the scalar (w_c*consistency +
    # w_a*accuracy - length_hinge); the full metric dict is kept alongside
    # for reporting and meta-LM visibility.
    def _log_prompt(tag: str, text: str) -> None:
        preview = text.strip().replace("\n", " ⏎ ")
        if len(preview) > 400:
            preview = preview[:400] + f"... [{len(text)} chars total]"
        logger.info("%s prompt: %s", tag, preview)

    # Each iteration runs evaluate_full_metrics on the subset — same number
    # of API calls as evaluate_position_bias (both do AB+BA per pair) but
    # returns consistency + accuracy + bias_rate, which feed the GEPA-style
    # score used to rank candidates.
    def _eval_fmt(m: dict) -> str:
        return (f"consistency={m['position_consistency']:.3f} "
                f"accuracy={m['accuracy']:.3f} "
                f"bias_rate={m['position_bias_rate']:.3f}")

    history = []
    for i, sp in enumerate(seed_prompts):
        m = await evaluate_full_metrics(client, eval_subset, sp,
                                        experiment_id=f"opro_seed_{i}")
        history.append({"prompt": sp, "score": compute_score(sp, m), "metrics": m})
        _log_prompt(f"Seed #{i}", sp)
        logger.info("Seed #%d  %s", i, _eval_fmt(m))

    # Main loop: generate candidate → evaluate on subset → add to history.
    # Each iteration logs BOTH the candidate prompt and its full metrics.
    # proposer_client (optional) lets the meta-LM that generates candidates
    # differ from the judge client (e.g. Llama judge + Qwen3-235B proposer).
    meta_client = proposer_client or client
    for it in range(n_iterations):
        logger.info("Iteration %d/%d", it + 1, n_iterations)
        candidate = await generate_candidate(meta_client, history)
        m = await evaluate_full_metrics(client, eval_subset, candidate,
                                        experiment_id=f"opro_iter_{it + 1}")
        history.append({"prompt": candidate, "score": compute_score(candidate, m), "metrics": m})
        _log_prompt(f"Iter {it + 1} candidate", candidate)
        logger.info("Iter %d  %s", it + 1, _eval_fmt(m))

    # Best across all attempts on the fast subset → promote to final eval.
    best_entry = max(history, key=lambda x: x["score"])
    logger.info("Best subset-consistency: %.3f", best_entry["score"])

    # Re-evaluate the winner on full train + val with the full metric
    # dict so it can be compared 1:1 against the GEPA result JSON.
    logger.info("Re-evaluating best prompt on full train and val (separate splits)")
    train = await evaluate_full_metrics(client, train_pairs, best_entry["prompt"], experiment_id="opro_best_train")
    val   = await evaluate_full_metrics(client, val_pairs,   best_entry["prompt"], experiment_id="opro_best_val")
    logger.info(
        "Best (train) consistency=%.3f accuracy=%.3f bias_rate=%.3f",
        train["position_consistency"], train["accuracy"], train["position_bias_rate"],
    )
    logger.info(
        "Best (val)   consistency=%.3f accuracy=%.3f bias_rate=%.3f",
        val["position_consistency"], val["accuracy"], val["position_bias_rate"],
    )

    result = OproResult(
        best_prompt=best_entry["prompt"],
        best_train_score=best_entry["score"],
        baseline_train=baseline_train,
        baseline_val=baseline_val,
        train=train,
        val=val,
        history=history,
    )

    with open(output_dir / "opro_results.json", "w") as f:
        json.dump({
            "best_prompt":       result.best_prompt,
            "best_train_score":  result.best_train_score,  # subset scalar (OPRO objective)
            "baseline_train":    result.baseline_train,    # full metrics, matches GEPA JSON
            "baseline_val":      result.baseline_val,
            "train":             result.train,
            "val":               result.val,
            "history":           result.history,
        }, f, indent=2)
    with open(output_dir / "optimised_prompt.md", "w", encoding="utf-8") as f:
        f.write(result.best_prompt)

    return result
