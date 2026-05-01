"""
compare_models.py
-----------------
Run GEPA + OPRO + OPRO-Tree on each model in a list, collect the baseline
and optimised metrics, dump a comparison table + plots, and save the best
prompt found by each (model, method) into a dedicated folder.

Usage
-----
    python -m src.prompt_optimizer.compare_models \
        --models meta-llama/Llama-3.3-70B-Instruct swiss-ai/Apertus-70B-Instruct-2509 \
        --reflection-model meta-llama/Llama-3.3-70B-Instruct \
        --output-dir results/comparison
"""

import argparse
import asyncio
import json
import logging
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from src.datasets.dataset import load_dataset_pairs, load_stratified_pairs
from src.core.inference_client import InferenceConfig, SwissAIClient
from src.runall_prompt_optimization.gepa_position_bias import run_gepa
from src.runall_prompt_optimization.opro_position_bias import run_opro, evaluate_full_metrics
from src.runall_prompt_optimization.opro_tree_position_bias import run_opro_tree
from src.runall_prompt_optimization.scoring import SEED_PROMPTS, compute_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit these defaults, or override any of them via CLI flags.
# Every key here is exposed as an argparse argument (see parse_args).
# ---------------------------------------------------------------------------

# Models on which to run all optimisers (one optimisation pass per model).
DEFAULT_MODELS = [
    "swiss-ai/Apertus-70B-Instruct-2509",
    "meta-llama/Llama-3.3-70B-Instruct",
]

# Reflection LM used by GEPA — kept fixed across all task models so GEPA
# results stay comparable. OPRO/OPRO-Tree don't use a separate reflection LM.
DEFAULT_REFLECTION_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Which optimisers to run. Drop any of these to skip that method.
DEFAULT_METHODS = ["gepa", "opro", "opro_tree"]

# I/O
DEFAULT_OUTPUT_DIR = Path("results/comparison")
DEFAULT_BASE_URL = "https://api.swissai.cscs.ch/v1"
DEFAULT_CONCURRENCY = 16  # shared by OPRO/OPRO-Tree (semaphore) and GEPA (num_threads)
DEFAULT_THINKING = "off"   # judge inference: 'off' | 'on'

# Dataset sizing + sampling
DEFAULT_N_TRAIN = 800
DEFAULT_N_VAL = 200
DEFAULT_STRATIFIED = True  # balanced easy/medium/hard mix; pass --no-stratified for random subsample

# GEPA budget
DEFAULT_BUDGET = "heavy"   # 'light' | 'medium' | 'heavy'

# OPRO / OPRO-Tree
DEFAULT_OPRO_N_ITERATIONS = 10
DEFAULT_OPRO_EVAL_PAIRS = 30
DEFAULT_BRANCHING_FACTOR = 5

# Reproducibility
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

METRIC_KEYS = ["position_consistency", "accuracy", "position_bias_rate"]


def _safe_model_tag(model: str) -> str:
    """Turn 'swiss-ai/Apertus-70B-Instruct-2509' into a filesystem-safe tag."""
    return model.replace("/", "_").replace(":", "_")


def _thinking_body(mode: str) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": mode == "on"}}


# ---------------------------------------------------------------------------
# Per-method runners. Each returns a dict:
#   {
#       "baseline_train": dict, "baseline_val": dict,
#       "train": dict, "val": dict,
#       "best_prompt": str,
#   }
# or None if the run failed (failure is logged + recorded as a row).
# ---------------------------------------------------------------------------

def run_gepa_one(
    *,
    model: str,
    reflection_model: str,
    api_base: str,
    api_key: str,
    train_pairs,
    val_pairs,
    output_dir: Path,
    auto_budget: str,
    eval_pairs: int,
    num_threads: int,
    seed: int,
    precomputed_baseline_train: dict | None = None,
    precomputed_baseline_val: dict | None = None,
) -> dict | None:
    try:
        result = run_gepa(
            model_name=model,
            api_base=api_base,
            api_key=api_key,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            reflection_model=reflection_model,
            auto_budget=auto_budget,
            eval_pairs=eval_pairs,
            seed=seed,
            num_threads=num_threads,
            output_dir=output_dir,
            precomputed_baseline_train=precomputed_baseline_train,
            precomputed_baseline_val=precomputed_baseline_val,
        )
        return {
            "baseline_train": result.baseline_train,
            "baseline_val":   result.baseline_val,
            "train":          result.train,
            "val":            result.val,
            "best_prompt":    result.optimised_instruction,
        }
    except Exception as e:
        logger.error("GEPA failed for %s: %s", model, e)
        traceback.print_exc()
        return None


async def run_opro_one(
    *,
    client: SwissAIClient,
    train_pairs,
    val_pairs,
    output_dir: Path,
    n_iterations: int,
    eval_pairs: int,
    seed: int,
    precomputed_baseline_train: dict | None = None,
    precomputed_baseline_val: dict | None = None,
    proposer_client: SwissAIClient | None = None,
) -> dict | None:
    try:
        result = await run_opro(
            client,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            n_iterations=n_iterations,
            eval_pairs=eval_pairs,
            seed=seed,
            output_dir=output_dir,
            precomputed_baseline_train=precomputed_baseline_train,
            precomputed_baseline_val=precomputed_baseline_val,
            proposer_client=proposer_client,
        )
        return {
            "baseline_train": result.baseline_train,
            "baseline_val":   result.baseline_val,
            "train":          result.train,
            "val":            result.val,
            "best_prompt":    result.best_prompt,
        }
    except Exception as e:
        logger.error("OPRO failed: %s", e)
        traceback.print_exc()
        return None


async def run_opro_tree_one(
    *,
    client: SwissAIClient,
    train_pairs,
    val_pairs,
    output_dir: Path,
    n_iterations: int,
    eval_pairs: int,
    branching_factor: int,
    seed: int,
    precomputed_baseline_train: dict | None = None,
    precomputed_baseline_val: dict | None = None,
    proposer_client: SwissAIClient | None = None,
) -> dict | None:
    try:
        result = await run_opro_tree(
            client,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            n_iterations=n_iterations,
            eval_pairs=eval_pairs,
            branching_factor=branching_factor,
            seed=seed,
            output_dir=output_dir,
            precomputed_baseline_train=precomputed_baseline_train,
            precomputed_baseline_val=precomputed_baseline_val,
            proposer_client=proposer_client,
        )
        return {
            "baseline_train": result.baseline_train,
            "baseline_val":   result.baseline_val,
            "train":          result.train,
            "val":            result.val,
            "best_prompt":    result.best_prompt,
        }
    except Exception as e:
        logger.error("OPRO-Tree failed: %s", e)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def rows_from_results(all_results: dict) -> list[dict]:
    """all_results[model][method] = {"baseline_val": {...}, "val": {...}, ...}"""
    rows = []
    for model, methods in all_results.items():
        for method, r in methods.items():
            if r is None:
                rows.append({
                    "model": model, "method": method, "split": "val", "stage": "FAILED",
                    **{k: None for k in METRIC_KEYS},
                })
                continue
            for split in ("train", "val"):
                for stage, key in [("baseline", f"baseline_{split}"), ("optimised", split)]:
                    m = r[key]
                    rows.append({
                        "model": model,
                        "method": method,
                        "split": split,
                        "stage": stage,
                        **{k: m.get(k) for k in METRIC_KEYS},
                    })
    return rows


def save_table(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results_long.csv", index=False)

    # Wide view: one row per (model, method), baseline vs optimised on val.
    val = df[df["split"] == "val"].copy()
    wide = val.pivot_table(
        index=["model", "method"],
        columns="stage",
        values=METRIC_KEYS,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{stage}" for metric, stage in wide.columns]
    wide = wide.reset_index()
    for k in METRIC_KEYS:
        if f"{k}_optimised" in wide.columns and f"{k}_baseline" in wide.columns:
            wide[f"delta_{k}"] = wide[f"{k}_optimised"] - wide[f"{k}_baseline"]
    wide.to_csv(out_dir / "results_wide_val.csv", index=False)

    with open(out_dir / "results_wide_val.md", "w") as f:
        f.write(wide.to_markdown(index=False, floatfmt=".3f"))

    # Summary: long-format table on val, baseline rows first then optimised.
    summary = val[val["stage"].isin(["baseline", "optimised"])].copy()
    summary = summary.sort_values(["model", "method", "stage"],
                                  key=lambda s: s.map({"baseline": 0, "optimised": 1})
                                  if s.name == "stage" else s)
    cols = ["model", "method", "stage"] + METRIC_KEYS
    summary = summary[cols]
    with open(out_dir / "results_summary.md", "w") as f:
        f.write("# Results (val split)\n\n")
        f.write(summary.to_markdown(index=False, floatfmt=".3f"))


def save_best_prompts(all_results: dict, out_dir: Path) -> None:
    """Write each (model, method) best_prompt as a .txt under <out_dir>/best_prompts/
    plus a single best_prompts.md index for quick review."""
    prompts_dir = out_dir / "best_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    md_lines = ["# Best prompts found per (model, method)", ""]
    for model, methods in all_results.items():
        tag = _safe_model_tag(model)
        md_lines.append(f"## `{model}`")
        md_lines.append("")
        for method, r in methods.items():
            md_lines.append(f"### {method}")
            md_lines.append("")
            if r is None:
                md_lines.append("_run failed_")
                md_lines.append("")
                continue
            prompt = r.get("best_prompt") or "(no prompt returned)"
            file_path = prompts_dir / f"{tag}__{method}.md"
            file_path.write_text(prompt)
            md_lines.append("```text")
            md_lines.append(prompt)
            md_lines.append("```")
            md_lines.append("")
            md_lines.append(f"_Saved to_: `{file_path.relative_to(out_dir)}`")
            md_lines.append("")
    (out_dir / "best_prompts.md").write_text("\n".join(md_lines))


def plot_per_metric(df: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    """For each metric, bar chart: x=model, bars grouped by method, value=val optimised."""
    val_opt = df[(df["split"] == "val") & (df["stage"] == "optimised")]
    val_base = df[(df["split"] == "val") & (df["stage"] == "baseline")]

    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRIC_KEYS:
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * val_opt["model"].nunique() * len(methods)), 4))
        piv = val_opt.pivot_table(
            index="model", columns="method", values=metric, aggfunc="first"
        )
        piv = piv.reindex(columns=[m for m in methods if m in piv.columns])
        piv.plot(kind="bar", ax=ax, width=0.75)

        base_by_model = val_base.groupby("model")[metric].first().reindex(piv.index)
        for i, (_, bval) in enumerate(base_by_model.items()):
            if pd.notna(bval):
                ax.hlines(bval, i - 0.4, i + 0.4,
                          colors="black", linestyles="--", linewidth=1)
        ax.set_title(f"{metric} on val (dashed = baseline)")
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.legend(title="method", loc="best")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(out_dir / f"plot_{metric}.png", dpi=150)
        plt.close(fig)


def plot_delta_consistency(df: pd.DataFrame, methods: list[str], out_dir: Path) -> None:
    val = df[df["split"] == "val"]
    piv_opt = val[val["stage"] == "optimised"].pivot_table(
        index="model", columns="method", values="position_consistency", aggfunc="first"
    )
    piv_base = val[val["stage"] == "baseline"].pivot_table(
        index="model", columns="method", values="position_consistency", aggfunc="first"
    )
    delta = (piv_opt - piv_base).reindex(columns=[m for m in methods if m in piv_opt.columns])

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * delta.shape[0] * len(methods)), 4))
    delta.plot(kind="bar", ax=ax, width=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Δ position_consistency (optimised − baseline) on val")
    ax.set_ylabel("Δ consistency")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "plot_delta_consistency.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_all_methods_for_model(
    *,
    model: str,
    reflection_model: str,
    api_base: str,
    api_key: str,
    train_pairs,
    val_pairs,
    out_dir: Path,
    methods: list[str],
    auto_budget: str,
    opro_n_iterations: int,
    opro_eval_pairs: int,
    opro_tree_branching_factor: int,
    concurrency: int,
    thinking: str,
    seed: int,
) -> dict:
    tag = _safe_model_tag(model)
    results: dict = {}

    # Compute shared baseline: evaluate both seed prompts on full train+val,
    # then pick the one with the higher val score as the common baseline.
    logger.info("[%s] Computing shared baseline (evaluating both seed prompts)", model)
    bl_cfg = InferenceConfig(
        base_url=api_base, api_key=api_key, model=model,
        concurrent_requests=concurrency,
    )
    async with SwissAIClient(bl_cfg) as bl_client:
        bt0 = await evaluate_full_metrics(bl_client, train_pairs, SEED_PROMPTS[0], experiment_id="shared_baseline_train_s0")
        bv0 = await evaluate_full_metrics(bl_client, val_pairs,   SEED_PROMPTS[0], experiment_id="shared_baseline_val_s0")
        bt1 = await evaluate_full_metrics(bl_client, train_pairs, SEED_PROMPTS[1], experiment_id="shared_baseline_train_s1")
        bv1 = await evaluate_full_metrics(bl_client, val_pairs,   SEED_PROMPTS[1], experiment_id="shared_baseline_val_s1")

    score0 = compute_score(SEED_PROMPTS[0], bv0)
    score1 = compute_score(SEED_PROMPTS[1], bv1)
    if score1 > score0:
        shared_baseline_train, shared_baseline_val = bt1, bv1
        logger.info("[%s] Baseline = seed 1 (val score=%.3f > seed 0 score=%.3f)", model, score1, score0)
    else:
        shared_baseline_train, shared_baseline_val = bt0, bv0
        logger.info("[%s] Baseline = seed 0 (val score=%.3f >= seed 1 score=%.3f)", model, score0, score1)
    logger.info(
        "[%s] Shared baseline — train cons=%.3f acc=%.3f | val cons=%.3f acc=%.3f",
        model,
        shared_baseline_train["position_consistency"], shared_baseline_train["accuracy"],
        shared_baseline_val["position_consistency"],   shared_baseline_val["accuracy"],
    )

    if "gepa" in methods:
        logger.info("[%s] Running GEPA (reflection=%s, budget=%s, threads=%d)",
                    model, reflection_model, auto_budget, concurrency)
        results["gepa"] = run_gepa_one(
            model=model,
            reflection_model=reflection_model,
            api_base=api_base,
            api_key=api_key,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            output_dir=out_dir / f"gepa_{tag}",
            auto_budget=auto_budget,
            eval_pairs=opro_eval_pairs,
            num_threads=concurrency,
            seed=seed,
            precomputed_baseline_train=shared_baseline_train,
            precomputed_baseline_val=shared_baseline_val,
        )

    if "opro" in methods or "opro_tree" in methods:
        cfg = InferenceConfig(
            base_url=api_base, api_key=api_key, model=model,
            concurrent_requests=concurrency,
            extra_body=_thinking_body(thinking),
        )
        proposer_cfg = InferenceConfig(
            base_url=api_base, api_key=api_key, model=reflection_model,
            concurrent_requests=concurrency,
        )
        async with SwissAIClient(cfg) as client, SwissAIClient(proposer_cfg) as proposer_client:
            if "opro" in methods:
                logger.info("[%s] Running OPRO (proposer=%s)", model, reflection_model)
                results["opro"] = await run_opro_one(
                    client=client,
                    train_pairs=train_pairs,
                    val_pairs=val_pairs,
                    output_dir=out_dir / f"opro_{tag}",
                    n_iterations=opro_n_iterations,
                    eval_pairs=opro_eval_pairs,
                    seed=seed,
                    precomputed_baseline_train=shared_baseline_train,
                    precomputed_baseline_val=shared_baseline_val,
                    proposer_client=proposer_client,
                )
            if "opro_tree" in methods:
                logger.info("[%s] Running OPRO-Tree (branching=%d, proposer=%s)",
                            model, opro_tree_branching_factor, reflection_model)
                results["opro_tree"] = await run_opro_tree_one(
                    client=client,
                    train_pairs=train_pairs,
                    val_pairs=val_pairs,
                    output_dir=out_dir / f"opro_tree_{tag}",
                    n_iterations=opro_n_iterations,
                    eval_pairs=opro_eval_pairs,
                    branching_factor=opro_tree_branching_factor,
                    seed=seed,
                    precomputed_baseline_train=shared_baseline_train,
                    precomputed_baseline_val=shared_baseline_val,
                    proposer_client=proposer_client,
                )
    return results


async def main_async(args):
    api_key = __import__("os").environ.get("SWISSAI_API_KEY", "")

    if args.stratified:
        logger.info("Loading stratified train/val (easy/medium/hard, ~n/3 per tier)")
        train_pairs = load_stratified_pairs(split="train", n=args.n_train, seed=args.seed)
        val_pairs   = load_stratified_pairs(split="validation", n=args.n_val, seed=args.seed)
    else:
        logger.info("Loading random subsample train/val")
        train_pairs = load_dataset_pairs(split="train", n=args.n_train, seed=args.seed)
        val_pairs   = load_dataset_pairs(split="validation", n=args.n_val, seed=args.seed)
    logger.info("Loaded %d train + %d val", len(train_pairs), len(val_pairs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    for model in args.models:
        logger.info("=" * 70)
        logger.info("Model: %s", model)
        logger.info("=" * 70)
        all_results[model] = await run_all_methods_for_model(
            model=model,
            reflection_model=args.reflection_model,
            api_base=args.base_url,
            api_key=api_key,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            out_dir=args.output_dir,
            methods=args.methods,
            auto_budget=args.budget,
            opro_n_iterations=args.opro_n_iterations,
            opro_eval_pairs=args.opro_eval_pairs,
            opro_tree_branching_factor=args.branching_factor,
            concurrency=args.concurrency,
            thinking=args.thinking,
            seed=args.seed,
        )

    # Persist raw results, table, plots, prompts.
    with open(args.output_dir / "raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    df = pd.DataFrame(rows_from_results(all_results))
    save_table(df, args.output_dir)
    plot_per_metric(df, args.methods, args.output_dir)
    plot_delta_consistency(df, args.methods, args.output_dir)
    save_best_prompts(all_results, args.output_dir)

    print("\n" + "=" * 70)
    print("Comparison summary (val, optimised vs baseline)")
    print("=" * 70)
    with open(args.output_dir / "results_wide_val.md") as f:
        print(f.read())
    print(f"\nArtifacts: {args.output_dir}")
    print(f"Best prompts saved under: {args.output_dir / 'best_prompts'}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run GEPA + OPRO + OPRO-Tree on a list of models and compare.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Models
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="One or more model IDs (Swiss AI Stack).")
    p.add_argument("--reflection-model", default=DEFAULT_REFLECTION_MODEL,
                   help="Reflection LM used by GEPA (kept fixed across all task models).")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   choices=["gepa", "opro", "opro_tree"],
                   help="Subset of optimisers to run.")

    # I/O
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--thinking", choices=["off", "on"], default=DEFAULT_THINKING)

    # Dataset
    p.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    p.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    p.add_argument("--stratified", action=argparse.BooleanOptionalAction,
                   default=DEFAULT_STRATIFIED,
                   help="Stratified easy/medium/hard mix; pass --no-stratified for random subsample.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # GEPA budget
    p.add_argument("--budget", choices=["light", "medium", "heavy"], default=DEFAULT_BUDGET,
                   help="GEPA auto budget.")

    # OPRO / OPRO-Tree
    p.add_argument("--opro-n-iterations", type=int, default=DEFAULT_OPRO_N_ITERATIONS,
                   help="Iterations for OPRO and OPRO-Tree.")
    p.add_argument("--opro-eval-pairs", type=int, default=DEFAULT_OPRO_EVAL_PAIRS,
                   help="Fast-eval subset size per OPRO/OPRO-Tree iteration.")
    p.add_argument("--branching-factor", type=int, default=DEFAULT_BRANCHING_FACTOR,
                   help="Candidates per OPRO-Tree iteration.")
    return p.parse_args()


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
