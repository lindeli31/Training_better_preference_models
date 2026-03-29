"""
opro_position_bias.py
---------------------
OPRO (Optimization by PROmpting) for reducing position bias in LLM judges.

The idea: use the LLM itself to iteratively generate better system prompts.
Each candidate prompt is scored by how consistently the judge gives the same
verdict when the order of responses A and B is swapped (position consistency).
The LLM sees the history of (prompt, score) pairs and tries to propose a
prompt that scores higher.

Pipeline:
  1. Start with 2 seed prompts (expert_rater, llm_judge)
  2. For N iterations: show history to LLM → generate candidate → evaluate it
  3. Pick the best-scoring prompt on training data
  4. Validate it on a held-out validation set
  5. Save everything to results/opro/opro_results.json
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from src.dataset import PairRecord
from src.inference_client import SwissAIClient
from src.templates import _build_user_prompt, _OUTPUT_INSTRUCTION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluate position consistency of a system prompt
# ---------------------------------------------------------------------------

async def evaluate_position_bias(
    client: SwissAIClient,
    pairs: list[PairRecord],
    system_prompt: str,
    criterion: str = "more helpful to the user",
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
    what works and what doesn't.
    """
    lines = ["Previous attempts (prompt → consistency score):\n"]
    for i, entry in enumerate(history, 1):
        lines.append(f"--- Attempt {i} (score: {entry['score']:.3f}) ---")
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
    best_prompt: str          # the system prompt with highest train score
    best_train_score: float   # its position consistency on the train subset
    val_score: float          # its position consistency on held-out validation
    history: list[dict]       # all attempts: [{"prompt": ..., "score": ...}, ...]


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

    # Sample a fixed subset of train pairs — used for all evaluations
    # so that scores are comparable across iterations
    rng = random.Random(seed)
    eval_subset = rng.sample(train_pairs, min(eval_pairs, len(train_pairs)))

    # Two seed prompts as starting points for the optimisation
    seed_prompts = [
        (
            "You are an expert rater. You will be provided with a user prompt and two "
            "assistant responses. Your task is to determine which response is better.\n\n"
            + _OUTPUT_INSTRUCTION
        ),
        (
            "You are an LLM used to judge and compare assistant completions. You will be "
            "given a user prompt and two assistant responses labeled A and B.\n\n"
            + _OUTPUT_INSTRUCTION
        ),
    ]

    # Evaluate seed prompts to initialise the history
    history = []
    for sp in seed_prompts:
        score = await evaluate_position_bias(client, eval_subset, sp)
        history.append({"prompt": sp, "score": score})
        logger.info("Seed prompt score: %.3f", score)

    # Main loop: generate candidate → evaluate → add to history
    # The LLM sees the full history each time, so it learns from all past attempts
    for it in range(n_iterations):
        logger.info("Iteration %d/%d", it + 1, n_iterations)
        candidate = await generate_candidate(client, history)
        score = await evaluate_position_bias(client, eval_subset, candidate)
        history.append({"prompt": candidate, "score": score})
        logger.info("Candidate score: %.3f", score)

    # Select the best prompt across all attempts (seeds + candidates)
    best_entry = max(history, key=lambda x: x["score"])
    logger.info("Best train score: %.3f", best_entry["score"])

    # Final validation on held-out data to check generalisation
    val_score = await evaluate_position_bias(client, val_pairs, best_entry["prompt"])
    logger.info("Validation score: %.3f", val_score)

    # Save everything
    result = OproResult(
        best_prompt=best_entry["prompt"],
        best_train_score=best_entry["score"],
        val_score=val_score,
        history=history,
    )

    with open(output_dir / "opro_results.json", "w") as f:
        json.dump({
            "best_prompt": result.best_prompt,
            "best_train_score": result.best_train_score,
            "val_score": result.val_score,
            "history": result.history,
        }, f, indent=2)

    logger.info("OPRO done. Train: %.3f | Validation: %.3f",
                result.best_train_score, result.val_score)

    return result
