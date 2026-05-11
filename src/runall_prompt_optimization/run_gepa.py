"""
run_gepa.py
-----------
CLI entry point for GEPA (DSPy) prompt optimisation targeting position bias.

Why GEPA on "hard" examples
    Unlike OPRO — which only shows the optimiser previous (prompt, score)
    pairs — GEPA reads TEXTUAL FEEDBACK explaining *why* each failure
    happened and performs a TARGETED mutation of the judge prompt.
    We feed it "hard" pairs (small score_gap) because that's where a
    biased judge falls back on positional cues: when the two responses
    are clearly different in quality, even a biased judge gets it right;
    when they're close, the bias dominates and GEPA has room to learn.

Output
    results/gepa/gepa_results.json — baseline / train / val metrics +
    the final evolved judge instruction. Numbers are directly
    comparable to the expert_rater baseline (same compute_position_bias).

Usage
-----
    python run_gepa.py                                       # defaults
    python run_gepa.py --n-train 100 --n-val 50 --budget medium
    python run_gepa.py --max-score-gap 0.5                   # easier pairs too
"""

# Standard library
import argparse
import logging
import os
from pathlib import Path

# Load SWISSAI_API_KEY from .env BEFORE importing modules that may read it.
from dotenv import load_dotenv
load_dotenv()

# Project imports
from src.dataset.dataset import load_dataset_pairs, load_stratified_pairs
from src.runall_prompt_optimization.gepa_position_bias import run_gepa

# Console logging at INFO so we see baseline / GEPA progress / final numbers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI arguments
# All knobs the user may want to tune without editing source. Defaults are
# chosen for a quick "light" run on Swiss AI Stack with Qwen-235B.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GEPA prompt optimisation for position bias")

    # Dataset sizing. Defaults bumped (50/25 → 800/200) because DSPy GEPA
    # itself recommends "smallest val, largest train" — 200 val is still
    # plenty for ±0.035 CI on a binary metric and leaves more room for
    # training diversity.
    p.add_argument("--n-train", type=int, default=800,
                   help="Number of training pairs (GEPA evolves on these)")
    p.add_argument("--n-val", type=int, default=200,
                   help="Number of validation pairs (held-out check)")
    p.add_argument("--max-score-gap", type=float, default=0.33,
                   help="Upper bound on |score_a - score_b| to qualify as 'hard'. "
                        "Smaller → harder → more position-bias signal.")
    p.add_argument("--sample-all", action="store_true",
                   help="Use a random sample from the full dataset instead of "
                        "filtering by score_gap (i.e. ignore --max-score-gap).")
    p.add_argument("--stratified", action="store_true",
                   help="Use a balanced mix across difficulty tiers "
                        "(easy / medium / hard, ~n/3 each). Best for GEPA: "
                        "hard pairs give consistency signal (accuracy is "
                        "near floor there), easy pairs give real accuracy "
                        "signal — the reflection LM sees both. Overrides "
                        "--sample-all and --max-score-gap.")

    # GEPA compute
    p.add_argument("--budget", choices=["light", "medium", "heavy"], default="light",
                   help="DSPy GEPA auto-budget preset (controls #candidate prompts). "
                        "Ignored when --max-metric-calls is set.")
    p.add_argument("--max-metric-calls", type=int, default=None,
                   help="Hard cap on total metric calls during GEPA.compile(). "
                        "Overrides --budget so you can stop early when the "
                        "trajectory plot shows a plateau (e.g. 300 or 500).")
    p.add_argument("--minibatch-size", type=int, default=30,
                   help="reflection_minibatch_size: how many examples GEPA "
                        "scores per candidate and shows the reflection LM. "
                        "DSPy default is 3; too small for binary metrics — "
                        "std dev per-candidate is ≈ sqrt(0.25/n), so:\n"
                        "  3  → ±0.58 (95%% CI)  — noise dominates\n"
                        "  10 → ±0.31             — wins < 0.1 invisible\n"
                        "  20 → ±0.22             — acceptable\n"
                        "  30 → ±0.18             — default, sharper accepts\n"
                        "  50 → ±0.14             — if you have budget\n"
                        "Scale --max-metric-calls proportionally to keep a "
                        "reasonable number of candidates explored.")
    p.add_argument("--selection-strategy", choices=["pareto", "current_best"],
                   default="pareto",
                   help="candidate_selection_strategy. 'pareto' keeps a "
                        "diverse frontier; 'current_best' greedy-follows the "
                        "best candidate so far.")
    p.add_argument("--reflection-temp", type=float, default=0.9,
                   help="Temperature for the reflection LM. 0.7 = similar "
                        "candidates across runs (Llama tends to converge); "
                        "0.9 = more exploration, useful when the reflection "
                        "LM is stuck proposing same-shape prompts.")
    p.add_argument("--seed-variant",
                   choices=["simple", "rater", "judge", "opro_best"],
                   default="judge",
                   help="Starting point for GEPA evolution. "
                        "'simple'/'rater' = SYSTEM_EXPERT_RATER seed (same as "
                        "OPRO's first seed). 'judge' = SYSTEM_LLM_JUDGE seed "
                        "(same as OPRO's second seed). 'opro_best' = reads "
                        "results/opro/opro_results.json so GEPA can refine "
                        "OPRO's result.")
    p.add_argument("--num-threads", type=int, default=16,
                   help="Parallel forward() calls during GEPA evaluation")
    p.add_argument("--thinking", choices=["off", "on", "auto"], default="off",
                   help="Whether the task + reflection LMs use their "
                        "reasoning/thought pass. "
                        "'off'  (default) → enable_thinking=False for every "
                        "model. Benchmark position bias of reasoning models "
                        "without their thought pass, comparable to "
                        "non-reasoning judges. "
                        "'on'   → enable_thinking=True for every model. "
                        "Measure a reasoning model at full capability. "
                        "'auto' → legacy behaviour (disable only for Qwen "
                        "by name, leave others at model default).")
    p.add_argument("--reflection-thinking",
                   choices=["off", "on", "auto", "inherit"], default="inherit",
                   help="Override --thinking for the reflection LM only. "
                        "'inherit' (default) reuses --thinking. Use "
                        "'on' with a reasoning reflection model while "
                        "keeping the judge at --thinking off, to isolate "
                        "the effect of a stronger optimiser.")

    # Reproducibility + I/O
    p.add_argument("--seed", type=int, default=42, help="Random seed for pair sampling")
    p.add_argument("--output-dir", type=Path, default=Path("results/gepa"),
                   help="Where gepa_results.json is written")

    # Model + endpoint (Swiss AI Stack is OpenAI-compatible → handled by LiteLLM)
    p.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct",
                   help="Task LM (the judge). Same name used by inference_client.")
    p.add_argument("--reflection-model", default=None,
                   help="Model that proposes prompt mutations (defaults to --model)")
    p.add_argument("--base-url", default="https://api.swissai.cscs.ch/v1",
                   help="OpenAI-compatible endpoint")

    return p.parse_args()


# ---------------------------------------------------------------------------
# main(): load data → run GEPA → print summary
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # API key comes from .env (loaded at import time above).
    api_key = os.environ.get("SWISSAI_API_KEY", "")

    # 1) Load pairs. Default = random subsample from the full dataset.
    #    --stratified → balanced mix across easy/medium/hard tiers.
    if args.stratified:
        logger.info("Loading stratified mix (easy/medium/hard, ~n/3 per tier)")
        train_pairs = load_stratified_pairs(split="train",      n=args.n_train, seed=args.seed)
        val_pairs   = load_stratified_pairs(split="validation", n=args.n_val,   seed=args.seed)
    else:
        logger.info("Loading random subsample from full dataset (default)")
        train_pairs = load_dataset_pairs(split="train",      n=args.n_train, seed=args.seed)
        val_pairs   = load_dataset_pairs(split="validation", n=args.n_val,   seed=args.seed)
    logger.info("Loaded %d train, %d val pairs", len(train_pairs), len(val_pairs))

    # 2) Run the pipeline: baseline eval → GEPA.compile() → re-eval on train+val.
    result = run_gepa(
        model_name=args.model,
        api_base=args.base_url,
        api_key=api_key,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        reflection_model=args.reflection_model,
        auto_budget=args.budget,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.minibatch_size,
        candidate_selection_strategy=args.selection_strategy,
        reflection_temp=args.reflection_temp,
        seed_variant=args.seed_variant,
        num_threads=args.num_threads,
        output_dir=args.output_dir,
        thinking=args.thinking,
        reflection_thinking=args.reflection_thinking,
    )

    # 3) Human-readable summary. The full JSON (incl. per-condition breakdown)
    #    is saved by run_gepa() to args.output_dir / "gepa_results.json".
    def _fmt(m: dict) -> str:
        return (f"consistency: {m['position_consistency']:.3f}  "
                f"accuracy: {m['accuracy']:.3f}  "
                f"bias_rate: {m['position_bias_rate']:.3f}")

    print(f"\n{'='*70}")
    print(f"GEPA Prompt Optimisation Results")
    print(f"{'='*70}")
    print(f"  Baseline (train) → {_fmt(result.baseline_train)}")
    print(f"  Baseline (val)   → {_fmt(result.baseline_val)}")
    print(f"  GEPA     (train) → {_fmt(result.train)}")
    print(f"  GEPA     (val)   → {_fmt(result.val)}")
    print(f"\nOptimised instruction:\n{result.optimised_instruction}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
