"""
gepa_position_bias.py
---------------------
GEPA (Reflective Prompt Evolution) for reducing position bias in LLM judges.

Gold label: A if score_a > score_b, B if reversed, C if equal.
Score = (helpfulness + correctness + coherence) / 3.0   (src/dataset.py)

Pipeline
  1. DSPy Signature with instruction = seed prompt (mutable by GEPA).
  2. PermutationJudge runs judge in AB and BA order per pair.
  3. gepa_metric returns (score, feedback); reflection LM proposes mutations.
  4. Re-evaluate optimised judge on train + val; save to results/gepa/.
"""

# Standard library
import hashlib
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import dspy
from src.datasets.dataset import PairRecord
from src.runall_prompt_optimization.gepa_plots import plot_baseline_vs_optimised, plot_training_trajectory
from src.core.inference_client import extract_label
from src.eval.metrics import compute_position_bias
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global metric state — reset at the start of each run_gepa() call.
# ---------------------------------------------------------------------------
_METRIC_HISTORY: list[dict] = []
_TRAIN_IDS: set[str] = set()
_VAL_IDS:   set[str] = set()
_BASELINE_VAL_CONSISTENCY: float | None = None
_BEAT_BASELINE_LOGGED: set[str] = set()
_BEAT_BASELINE_MIN_VAL_CALLS = 50

from src.runall_prompt_optimization.scoring import compute_score, _FLIP, SEED_PROMPTS

# Guardrails shown to the reflection LM on every feedback call. Built once
# at module load so repeated minibatch calls don't reconstruct the string.
_GUARDRAILS = (
    "\n\n[Guidance] You are improving a system prompt for an LLM judge that "
    "compares two responses (A and B). The goal is to MINIMISE POSITION BIAS: "
    "the judge must give the same verdict regardless of which response appears first. "
    "Output ONLY the new system prompt text, nothing else."
)


# ---------------------------------------------------------------------------
# DSPy Signature and PermutationJudge module.
# ---------------------------------------------------------------------------

class _JudgeBase(dspy.Signature):
    """placeholder — instruction replaced via with_instructions()"""
    user_prompt: str = dspy.InputField(desc = "Messages in conversation history")
    response_a:  str = dspy.InputField(desc="First possible answer")
    response_b:  str = dspy.InputField(desc="Second possible answer")
    preference:  str = dspy.OutputField()


def get_seed_prompt(variant: str = "simple") -> str:
    if variant in {"simple", "rater"}:
        return SEED_PROMPTS[0]
    if variant == "judge":
        return SEED_PROMPTS[1]
    if variant == "opro_best":
        opro_path = Path("results/opro/opro_results.json")
        if not opro_path.exists():
            raise FileNotFoundError(f"{opro_path} not found — run OPRO first.")
        with open(opro_path) as f:
            data = json.load(f)
        return data["best_prompt"]
    raise ValueError(f"Unknown seed variant: {variant!r}")


def build_judge_signature(seed_variant: str = "simple"):
    return _JudgeBase.with_instructions(get_seed_prompt(seed_variant))


JudgeSignature = build_judge_signature("simple")


def _parse(raw) -> str:
    if not isinstance(raw, str) or not raw:
        return ""
    label, _ok = extract_label(raw)
    return label or ""


class PermutationJudge(dspy.Module):
    """Runs the judge in AB and BA order; both labels measure position consistency."""

    def __init__(self, signature=None):
        super().__init__()
        self.judge = dspy.Predict(signature or JudgeSignature)

    def forward(self, user_prompt, response_a, response_b):
        ab = self.judge(user_prompt=user_prompt, response_a=response_a, response_b=response_b)
        ba = self.judge(user_prompt=user_prompt, response_a=response_b, response_b=response_a)
        # Carry instruction on Prediction so the metric can apply penalties.
        return dspy.Prediction(
            ab_label=_parse(ab.preference),
            ba_label=_parse(ba.preference),
            instruction=self.judge.signature.instructions or "",
        )

# ---------------------------------------------------------------------------
# Metric helpers.
# ---------------------------------------------------------------------------

def _classify_failure(ab: str, ba: str, consistent: bool, accurate: bool) -> str:
    if consistent and accurate: return "OK"
    if ab == ba == "A":         return "ANCHORED_FIRST"
    if ab == ba == "B":         return "ANCHORED_SECOND"
    if not consistent:          return "RANDOM_FLIP"
    return "CONTENT_WRONG"

# ---------------------------------------------------------------------------
# GEPA metric — score + per-example feedback for the reflection LM.
# ---------------------------------------------------------------------------

def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    ab, ba, gold_label = pred.ab_label, pred.ba_label, gold.gold_label
    difficulty = getattr(gold, "difficulty", None)
    score_gap  = getattr(gold, "score_gap",  None)

    parse_ok  = (ab in {"A", "B", "C"}) and (ba in {"A", "B", "C"})
    consistent = (ba == _FLIP.get(ab))
    accurate   = (ab == gold_label)

    instruction = getattr(pred, "instruction", "")
    score = compute_score(instruction, {"position_consistency": int(consistent), "accuracy": int(accurate)})

    if instruction:
        pid   = getattr(gold, "prompt_id", None)
        split = "train" if pid in _TRAIN_IDS else ("val" if pid in _VAL_IDS else "unknown")
        _METRIC_HISTORY.append({
            "instruction_hash": hashlib.md5(instruction.encode()).hexdigest()[:8],
            "instruction_len":  len(instruction),
            "consistent":       int(consistent),
            "accurate":         int(accurate),
            "parse_ok":         int(parse_ok),
            "score":            float(score),
            "split":            split,
            "difficulty":       difficulty,
            "score_gap":        score_gap,
            "ab_is_c":          int(ab == "C"),
            "gold_is_c":        int(gold_label == "C"),
        })

        if len(_METRIC_HISTORY) % 10 == 0:
            recent   = _METRIC_HISTORY[-10:]
            m_score  = sum(e["score"]     for e in recent) / 10
            m_acc    = sum(e["accurate"]  for e in recent) / 10
            m_cons   = sum(e["consistent"]for e in recent) / 10
            m_parse  = sum(e["parse_ok"]  for e in recent) / 10
            m_c_emit = sum(e["ab_is_c"]   for e in recent) / 10
            m_c_gold = sum(e["gold_is_c"] for e in recent) / 10
            log_fn   = logger.warning if m_parse < 1.0 else logger.info
            log_fn(
                "GEPA progress: %d calls | score=%.3f acc=%.3f consist=%.3f "
                "parse=%.2f c_emit=%.2f c_gold=%.2f",
                len(_METRIC_HISTORY), m_score, m_acc, m_cons,
                m_parse, m_c_emit, m_c_gold,
            )

        # Live banner when a candidate first beats the baseline on val.
        if split == "val" and _BASELINE_VAL_CONSISTENCY is not None:
            h = hashlib.md5(instruction.encode()).hexdigest()[:8]
            if h not in _BEAT_BASELINE_LOGGED:
                cand_val = [e for e in _METRIC_HISTORY
                            if e["split"] == "val" and e["instruction_hash"] == h]
                if len(cand_val) >= _BEAT_BASELINE_MIN_VAL_CALLS:
                    cons = sum(e["consistent"] for e in cand_val) / len(cand_val)
                    if cons > _BASELINE_VAL_CONSISTENCY:
                        _BEAT_BASELINE_LOGGED.add(h)
                        acc = sum(e["accurate"] for e in cand_val) / len(cand_val)
                        logger.info(
                            "*** BEAT BASELINE on val *** candidate=%s "
                            "consistency=%.3f (> baseline %.3f) accuracy=%.3f n=%d",
                            h, cons, _BASELINE_VAL_CONSISTENCY, acc, len(cand_val),
                        )

    # Build per-example feedback for the reflection LM. Inputs (user_prompt,
    # response_a, response_b) are passed natively by GEPA, so no need to
    # duplicate them in the feedback string.
    mode        = _classify_failure(ab, ba, consistent, accurate)
    expected_ba = _FLIP.get(gold_label, "?")

    lines = [
        f"Metrics: consistency={int(consistent)} accuracy={int(accurate)} bias={int(not consistent)}",
        f"AB: judge={ab or '∅'} gold={gold_label} → {'✓' if ab == gold_label else '✗'}",
        f"BA: judge={ba or '∅'} expected={expected_ba} → {'✓' if ba == expected_ba else '✗'}",
        f"Failure: {mode}",
    ]

    return dspy.Prediction(score=score, feedback="\n".join(lines) + _GUARDRAILS)

# ---------------------------------------------------------------------------
# Data helpers.
# ---------------------------------------------------------------------------

def pairs_to_examples(pairs: list[PairRecord]) -> list[dspy.Example]:
    return [
        dspy.Example(
            user_prompt=p.prompt,
            response_a=p.response_a,
            response_b=p.response_b,
            gold_label=p.gold_label,
            prompt_id=p.prompt_id,
            difficulty=p.difficulty,
            score_gap=getattr(p, "score_gap", None),
        ).with_inputs("user_prompt", "response_a", "response_b")
        for p in pairs
    ]


def evaluate_judge(judge: PermutationJudge, pairs: list[PairRecord], num_threads: int = 8) -> dict:
    gold_by_pid = {p.prompt_id: p.gold_label for p in pairs}
    records = []
    ab_correct = ab_total = ba_correct = ba_total = 0

    def _one(p: PairRecord):
        try:
            out = judge(user_prompt=p.prompt, response_a=p.response_a, response_b=p.response_b)
            return p.prompt_id, out.ab_label, out.ba_label, None
        except Exception as e:
            return p.prompt_id, None, None, e

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        for pid, ab, ba, err in ex.map(_one, pairs):
            if err is not None:
                logger.warning("Judge failed on %s: %s", pid, err)
                continue
            gold = gold_by_pid[pid]
            records.append({"prompt_id": pid, "condition": "AB", "label": ab or None})
            records.append({"prompt_id": pid, "condition": "BA", "label": ba or None})
            if ab: ab_total += 1; ab_correct += int(ab == gold)
            if ba: ba_total += 1; ba_correct += int(ba == _FLIP[gold])

    metrics = compute_position_bias(records)
    metrics["accuracy_ab"] = round(ab_correct / ab_total, 4) if ab_total else 0.0
    metrics["accuracy_ba"] = round(ba_correct / ba_total, 4) if ba_total else 0.0
    total = ab_total + ba_total
    metrics["accuracy"]    = round((ab_correct + ba_correct) / total, 4) if total else 0.0
    return metrics

# ---------------------------------------------------------------------------
# Result container and main pipeline.
# ---------------------------------------------------------------------------

class GepaResult:
    def __init__(self, baseline_train, baseline_val, train, val, optimised_instruction):
        self.baseline_train       = baseline_train
        self.baseline_val         = baseline_val
        self.train                = train
        self.val                  = val
        self.optimised_instruction = optimised_instruction


def run_gepa(
    model_name: str,
    api_base: str,
    api_key: str,
    train_pairs: list[PairRecord],
    val_pairs: list[PairRecord],
    reflection_model: str | None = None,
    auto_budget: str = "light",
    max_metric_calls: int | None = None,
    eval_pairs: int = 30,
    seed: int = 42,
    reflection_minibatch_size: int = 30,
    candidate_selection_strategy: str = "pareto",
    reflection_temp: float = 0.9,
    num_threads: int = 8,
    output_dir: Path = Path("results/gepa"),
    thinking: str = "off",
    reflection_thinking: str = "inherit",
) -> GepaResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    if thinking not in {"off", "on", "auto"}:
        raise ValueError(f"thinking must be off/on/auto, got {thinking!r}")
    if reflection_thinking not in {"off", "on", "auto", "inherit"}:
        raise ValueError(f"reflection_thinking must be off/on/auto/inherit, got {reflection_thinking!r}")

    def _extra_body_for(mode: str, name: str) -> dict:
        if mode == "off": return {"chat_template_kwargs": {"enable_thinking": False}}
        if mode == "on":  return {"chat_template_kwargs": {"enable_thinking": True}}
        return {"chat_template_kwargs": {"enable_thinking": False}} if "Qwen" in name else {}

    refl_mode = thinking if reflection_thinking == "inherit" else reflection_thinking
    logger.info("thinking=%s reflection_thinking=%s", thinking, refl_mode)

    task_lm = dspy.LM(
        model=f"openai/{model_name}", api_base=api_base, api_key=api_key,
        temperature=0.0, max_tokens=2048,
        extra_body=_extra_body_for(thinking, model_name),
    )
    refl_name = reflection_model or model_name
    reflection_lm = dspy.LM(
        model=f"openai/{refl_name}", api_base=api_base, api_key=api_key,
        temperature=reflection_temp, max_tokens=1200,
        extra_body=_extra_body_for(refl_mode, refl_name),
    )
    dspy.configure(lm=task_lm)

    rng = random.Random(seed)
    eval_subset = rng.sample(train_pairs, min(eval_pairs, len(train_pairs)))
    trainset = pairs_to_examples(eval_subset)
    valset   = pairs_to_examples(val_pairs)

    # Baseline: always SEED_PROMPTS[0] (same as OPRO).
    logger.info("Evaluating baseline on %d train + %d val pairs", len(train_pairs), len(val_pairs))
    baseline_sig   = _JudgeBase.with_instructions(SEED_PROMPTS[0])
    baseline_judge = PermutationJudge(baseline_sig)
    baseline_train = evaluate_judge(baseline_judge, train_pairs, num_threads)
    logger.info("Baseline(train) cons=%.3f acc=%.3f bias=%.3f",
                baseline_train["position_consistency"], baseline_train["accuracy"],
                baseline_train["position_bias_rate"])
    baseline_val = evaluate_judge(baseline_judge, val_pairs, num_threads)
    logger.info("Baseline(val)   cons=%.3f acc=%.3f bias=%.3f",
                baseline_val["position_consistency"], baseline_val["accuracy"],
                baseline_val["position_bias_rate"])

    gepa_common = dict(
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        num_threads=num_threads,
        reflection_minibatch_size=reflection_minibatch_size,
        candidate_selection_strategy=candidate_selection_strategy,
        track_stats=True,
    )

    def _optimise_one_seed(seed_idx: int) -> tuple:
        """Run one GEPA optimisation from a given seed. Returns (judge, train, val, history)."""
        global _BASELINE_VAL_CONSISTENCY
        _METRIC_HISTORY.clear()
        _TRAIN_IDS.clear()
        _VAL_IDS.clear()
        _BEAT_BASELINE_LOGGED.clear()
        _BASELINE_VAL_CONSISTENCY = baseline_val["position_consistency"]
        _TRAIN_IDS.update(p.prompt_id for p in train_pairs)
        _VAL_IDS.update(p.prompt_id for p in val_pairs)

        sig = _JudgeBase.with_instructions(SEED_PROMPTS[seed_idx])
        logger.info("--- GEPA seed %d (%d chars) ---", seed_idx, len(SEED_PROMPTS[seed_idx]))

        if max_metric_calls is not None:
            gepa = dspy.GEPA(max_metric_calls=max_metric_calls, **gepa_common)
        else:
            gepa = dspy.GEPA(auto=auto_budget, **gepa_common)

        opt_judge = gepa.compile(
            student=PermutationJudge(sig),
            trainset=trainset,
            valset=valset,
        )
        t = evaluate_judge(opt_judge, train_pairs, num_threads)
        v = evaluate_judge(opt_judge, val_pairs,   num_threads)
        history = list(_METRIC_HISTORY)
        logger.info("Seed %d result: train cons=%.3f acc=%.3f | val cons=%.3f acc=%.3f",
                    seed_idx, t["position_consistency"], t["accuracy"],
                    v["position_consistency"], v["accuracy"])
        return opt_judge, t, v, history

    judge0, train0, val0, hist0 = _optimise_one_seed(0)
    judge1, train1, val1, hist1 = _optimise_one_seed(1)

    # Pick winner by val consistency; break ties with val accuracy.
    if (val0["position_consistency"], val0["accuracy"]) >= (val1["position_consistency"], val1["accuracy"]):
        winning_seed = 0
        opt_judge, train, val, winning_history = judge0, train0, val0, hist0
    else:
        winning_seed = 1
        opt_judge, train, val, winning_history = judge1, train1, val1, hist1

    logger.info("Winner: seed %d (val cons=%.3f acc=%.3f)", winning_seed,
                val["position_consistency"], val["accuracy"])

    result = GepaResult(
        baseline_train=baseline_train,
        baseline_val=baseline_val,
        train=train,
        val=val,
        optimised_instruction=opt_judge.judge.signature.instructions,
    )

    unique_hashes = {e["instruction_hash"] for e in winning_history}
    logger.info("GEPA metric history (winner): %d entries, %d unique candidates",
                len(winning_history), len(unique_hashes))

    all_history = [{"seed": 0, **e} for e in hist0] + [{"seed": 1, **e} for e in hist1]

    with open(output_dir / "gepa_results.json", "w") as f:
        json.dump({
            "baseline_train":        result.baseline_train,
            "baseline_val":          result.baseline_val,
            "train":                 result.train,
            "val":                   result.val,
            "optimised_instruction": result.optimised_instruction,
            "winning_seed":          winning_seed,
            "seed_0_prompt":         judge0.judge.signature.instructions,
            "seed_1_prompt":         judge1.judge.signature.instructions,
            "seed_0_val":            val0,
            "seed_1_val":            val1,
            "n_unique_candidates":   len(unique_hashes),
            "n_metric_calls":        len(all_history),
        }, f, indent=2)
    with open(output_dir / "gepa_metric_history.json", "w") as f:
        json.dump(all_history, f, indent=2)
    with open(output_dir / "optimised_prompt.md", "w", encoding="utf-8") as f:
        f.write(result.optimised_instruction)
    with open(output_dir / "seed_0_prompt.md", "w", encoding="utf-8") as f:
        f.write(judge0.judge.signature.instructions)
    with open(output_dir / "seed_1_prompt.md", "w", encoding="utf-8") as f:
        f.write(judge1.judge.signature.instructions)

    plot_training_trajectory(winning_history, output_dir)
    plot_baseline_vs_optimised(baseline_train, baseline_val, train, val, output_dir)
    logger.info("Plots saved to %s/plots/", output_dir)

    logger.info(
        "GEPA done. Baseline(val) cons=%.3f acc=%.3f | "
        "Seed0(val) cons=%.3f acc=%.3f | Seed1(val) cons=%.3f acc=%.3f | "
        "Winner=seed%d cons=%.3f acc=%.3f",
        baseline_val["position_consistency"], baseline_val["accuracy"],
        val0["position_consistency"], val0["accuracy"],
        val1["position_consistency"], val1["accuracy"],
        winning_seed, val["position_consistency"], val["accuracy"],
    )
    return result
