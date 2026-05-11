"""
update_baseline.py
------------------
Evaluate both seed prompts (expert_rater and llm_judge) on val for each
model, then regenerate results tables and plots with a compact layout:

  - Baseline rows appear ONCE per model (one row per seed prompt).
  - Optimised rows appear once per (model, method) — no baseline repetition.

Usage
-----
    python update_baseline.py \
        --raw-results comparison/raw_results.json \
        --output-dir  comparison \
        [--n-val 200] [--seed 42] \
        [--base-url https://api.swissai.cscs.ch/v1] \
        [--concurrency 16]
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.core.inference_client import InferenceConfig, SwissAIClient
from src.core.templates import SYSTEM_EXPERT_RATER, SYSTEM_LLM_JUDGE
from src.dataset.dataset import load_stratified_pairs
from src.runall_prompt_optimization.opro_position_bias import evaluate_full_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

METRIC_KEYS = ["position_consistency", "accuracy", "position_bias_rate"]

SEED_PROMPTS = [
    ("expert_rater", SYSTEM_EXPERT_RATER),
    ("llm_judge",    SYSTEM_LLM_JUDGE),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate_seeds_for_model(
    *,
    model: str,
    api_base: str,
    api_key: str,
    val_pairs,
    concurrency: int,
) -> dict[str, dict]:
    """Return {seed_name: val_metrics} for all seeds."""
    cfg = InferenceConfig(
        base_url=api_base, api_key=api_key,
        model=model, concurrent_requests=concurrency,
    )
    results = {}
    async with SwissAIClient(cfg) as client:
        for name, prompt in SEED_PROMPTS:
            logger.info("[%s] Evaluating seed '%s' on %d val pairs", model, name, len(val_pairs))
            metrics = await evaluate_full_metrics(
                client, val_pairs, prompt,
                experiment_id=f"update_baseline_{name}_val",
            )
            results[name] = metrics
            logger.info(
                "[%s] seed=%-15s | cons=%.3f  acc=%.3f  bias=%.3f",
                model, name,
                metrics["position_consistency"],
                metrics["accuracy"],
                metrics["position_bias_rate"],
            )
    return results


# ---------------------------------------------------------------------------
# Compact table builder
# ---------------------------------------------------------------------------

def build_compact_df(raw: dict, seed_results: dict[str, dict]) -> pd.DataFrame:
    """
    Two kinds of rows:
      - stage='baseline (expert_rater)' / 'baseline (llm_judge)': one per model, method='-'
      - stage='optimised': one per (model, method)
    """
    rows = []

    for model in raw:
        # Baseline rows — one per seed, shared across all methods.
        for seed_name, metrics in seed_results[model].items():
            rows.append({
                "model":  model,
                "method": "-",
                "stage":  f"baseline ({seed_name})",
                **{k: round(metrics.get(k, float("nan")), 3) for k in METRIC_KEYS},
            })

        # Optimised rows — one per method.
        for method, r in raw[model].items():
            if r is None:
                rows.append({
                    "model": model, "method": method, "stage": "optimised (FAILED)",
                    **{k: None for k in METRIC_KEYS},
                })
            else:
                m = r["val"]
                rows.append({
                    "model":  model,
                    "method": method,
                    "stage":  "optimised",
                    **{k: round(m.get(k, float("nan")), 3) for k in METRIC_KEYS},
                })

    return pd.DataFrame(rows)


def save_compact_table(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results_compact.csv", index=False)
    with open(out_dir / "results_compact.md", "w") as f:
        f.write("# Results (val split)\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
    logger.info("Compact table saved to %s", out_dir / "results_compact.md")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_compact(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models   = [m for m in df["model"].unique() if m != "-"]
    methods  = [m for m in df["method"].unique() if m != "-"]
    baseline_seeds = [s for s in df["stage"].unique() if s.startswith("baseline")]

    for metric in METRIC_KEYS:
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(models) * len(methods)), 4))

        opt = df[df["stage"] == "optimised"].pivot_table(
            index="model", columns="method", values=metric, aggfunc="first"
        ).reindex(index=models, columns=[m for m in methods if m in df["method"].values])
        opt.plot(kind="bar", ax=ax, width=0.7)

        # One dashed horizontal line per seed, per model.
        seed_styles = {"baseline (expert_rater)": ("--", "black"),
                       "baseline (llm_judge)":    (":",  "dimgray")}
        for i, model in enumerate(opt.index):
            for stage, (ls, color) in seed_styles.items():
                row = df[(df["model"] == model) & (df["stage"] == stage)]
                if not row.empty:
                    val = row[metric].values[0]
                    if pd.notna(val):
                        ax.hlines(val, i - 0.45, i + 0.45,
                                  colors=color, linestyles=ls, linewidth=1.4,
                                  label=stage if i == 0 else "_nolegend_")

        ax.set_title(f"{metric} on val  (-- expert_rater baseline  ··· llm_judge baseline)")
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="method / baseline", loc="best", fontsize=8)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(out_dir / f"plot_{metric}.png", dpi=150)
        plt.close(fig)
        logger.info("Saved plot_%s.png", metric)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get("SWISSAI_API_KEY", "")

    with open(args.raw_results) as f:
        raw: dict = json.load(f)
    models = list(raw.keys())
    logger.info("Models: %s", models)

    logger.info("Loading val pairs (n=%d, seed=%d)", args.n_val, args.seed)
    val_pairs = load_stratified_pairs(split="validation", n=args.n_val, seed=args.seed)
    logger.info("Loaded %d val pairs", len(val_pairs))

    seed_results: dict[str, dict] = {}
    for model in models:
        logger.info("=" * 60)
        logger.info("Model: %s", model)
        seed_results[model] = await evaluate_seeds_for_model(
            model=model,
            api_base=args.base_url,
            api_key=api_key,
            val_pairs=val_pairs,
            concurrency=args.concurrency,
        )

    # Build and save compact table.
    df = build_compact_df(raw, seed_results)
    save_compact_table(df, args.output_dir)
    plot_compact(df, args.output_dir)

    print("\n" + "=" * 70)
    print("Compact results (val split)")
    print("=" * 70)
    with open(args.output_dir / "results_compact.md") as f:
        print(f.read())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate both seed prompts and produce a compact results table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-results", type=Path, default=Path("comparison/raw_results.json"))
    p.add_argument("--output-dir",  type=Path, default=Path("comparison"))
    p.add_argument("--base-url",    default="https://api.swissai.cscs.ch/v1")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--n-val",       type=int, default=200)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
