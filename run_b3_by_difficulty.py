"""
run_b3_by_difficulty.py
-----------------------
B3 Reasoning Depth: Position bias evaluation across templates and difficulty buckets.

For a given difficulty bucket (easy/medium/hard), judge each pair TWICE in both orders
(AB and BA) with THREE different templates:
  1. expert_rater (baseline, no reasoning)
  2. reason_then_judge (explain reasoning, then verdict)
  3. structured_reasoning (rate on criteria, then verdict)

This isolates the effect of reasoning depth on position bias within each difficulty level.

Usage
-----
    python run_b3_by_difficulty.py --difficulty easy --n-pairs 300
    python run_b3_by_difficulty.py --difficulty hard --n-pairs 500 --run-name hard_reasoning

Output: results/b3_by_difficulty/<difficulty>_<template>_<run_name>.jsonl
         + .meta.json sidecar

Each run produces position bias metrics (consistency, primacy, recency, tie inconsistency,
accuracy, accuracy_gap) for one template on one difficulty bucket.
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from src.core.inference_client import InferenceConfig, SwissAIClient
from src.core.templates import TEMPLATES
from src.datasets.dataset import DIFFICULTY_LEVELS, load_dataset_pairs
from src.eval.experiments import run_position_bias
from src.eval.metrics import compute_position_bias, print_summary
from check_models import validate_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# B3 templates to compare
REASONING_TEMPLATES = [
    ("expert_rater", "No reasoning (baseline)"),
    ("reason_then_judge", "Explain reasoning, then verdict"),
    ("structured_reasoning", "Rate on criteria, then verdict"),
    ("opro_llama", "OPRO-best for Llama (letter-only output)"),
    ("opro_llama_reason_then_judge", "OPRO-best for Llama + reason-then-judge"),
    ("tree_of_thoughts_judge", "Tree-of-Thoughts: brainstorm/evaluate/prune/verdict"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="B3: Position bias across reasoning templates and difficulty buckets"
    )
    p.add_argument("--difficulty", required=True, choices=list(DIFFICULTY_LEVELS),
                   help="Which difficulty bucket to evaluate")
    p.add_argument("--n-pairs", type=int, default=300,
                   help="Number of pairs to evaluate")
    p.add_argument("--split", default="train",
                   help="Dataset split (train / validation)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for dataset shuffle")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0.0 = greedy)")
    p.add_argument("--criterion", default="overall",
                   help="Evaluation criterion")
    p.add_argument("--run-name", default=None,
                   help="Short label appended to output filename (default: UTC timestamp)")
    p.add_argument("--comment", default="",
                   help="Free-text comment saved in sidecar .meta.json")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--experiment-name", default="b3_by_difficulty",
                   help="Subdirectory under --output-dir where results are saved "
                        "(use 'b4_by_difficulty' for the B4 language sweep).")
    p.add_argument("--model", default=os.environ.get("SWISSAI_MODEL", "meta-llama/Llama-3.3-70B-Instruct"))
    p.add_argument("--base-url", default="https://api.swissai.cscs.ch/v1")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--exclude-ties", action="store_true",
                   help="Exclude pairs with gold_label=C from accuracy computation")
    p.add_argument("--templates", nargs="+", default=None,
                   help="Subset of templates to run. Defaults to all entries "
                        "in REASONING_TEMPLATES. Any registered template id is "
                        "accepted (e.g. tree_of_thoughts_judge or expert_rater_pl/de/it "
                        "for the B4 language sweep).")
    return p.parse_args()


async def main(args):
    api_key = os.environ.get("SWISSAI_API_KEY", "")
    if not api_key:
        logger.warning("SWISSAI_API_KEY not set. Requests may fail unless the endpoint is open.")
    validate_model(args.model, base_url=args.base_url, api_key=api_key)

    config = InferenceConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        concurrent_requests=args.concurrency,
    )

    logger.info(
        "Loading pairs: split=%s | difficulty=%s | n=%d | seed=%d",
        args.split, args.difficulty, args.n_pairs, args.seed
    )
    pairs = load_dataset_pairs(
        split=args.split,
        n=args.n_pairs,
        seed=args.seed,
        difficulty=args.difficulty,
        full=True,
    )
    logger.info("Pairs loaded: %d", len(pairs))

    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if args.templates:
        # Accept any registered template id, even if it's outside REASONING_TEMPLATES
        # (e.g. the B4 language variants expert_rater_pl/de/it).
        known_desc = dict(REASONING_TEMPLATES)
        templates_to_run = [
            (t, known_desc.get(t, TEMPLATES[t].get("description", t)))
            for t in args.templates
        ]
    else:
        templates_to_run = REASONING_TEMPLATES

    # Run B1 (position bias) for each reasoning template
    all_metrics = {}
    async with SwissAIClient(config) as client:
        for template_id, template_desc in templates_to_run:
            logger.info(
                "=== B3 Evaluation (difficulty=%s, template=%s, run=%s) ===",
                args.difficulty, template_id, run_name
            )
            logger.info("Template description: %s", template_desc)

            # Run position bias experiment (AB + BA) with this template
            t_start = time.perf_counter()
            results = await run_position_bias(
                client,
                pairs,
                template_id=template_id,
                criterion=args.criterion,
                output_dir=args.output_dir / args.experiment_name / args.difficulty,
            )
            elapsed = time.perf_counter() - t_start

            # Build gold labels for accuracy computation
            gold_labels = {p.prompt_id: p.gold_label for p in pairs}

            # Compute metrics
            results_dicts = [r.to_dict() for r in results]
            metrics = compute_position_bias(results_dicts, gold_labels=gold_labels, exclude_ties=args.exclude_ties)

            # Store metrics for comparison
            all_metrics[template_id] = metrics

            # Print summary for this template
            print_summary(
                f"B3 Results (difficulty={args.difficulty}, template={template_id}, run={run_name})",
                metrics
            )
            logger.info("Template completed in %.1f s", elapsed)

    # Comparison table: print metrics across all templates
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Position bias metrics across reasoning templates")
    logger.info("Difficulty bucket: %s | Run: %s", args.difficulty, run_name)
    logger.info("=" * 80)

    template_labels = {
        "expert_rater": "No Reasoning",
        "reason_then_judge": "Reason→Judge",
        "structured_reasoning": "Structured",
        "opro_llama": "OPRO-Llama",
        "opro_llama_reason_then_judge": "OPRO+Reason",
        "tree_of_thoughts_judge": "Tree-of-Thoughts",
        "expert_rater_pl": "Polish",
        "expert_rater_de": "German",
        "expert_rater_it": "Italian",
    }
    header = ["Metric"] + [template_labels.get(t, t) for t, _ in templates_to_run]
    print("\n" + "".join(f"{c:<30}" if i == 0 else f"{c:<18}" for i, c in enumerate(header)))
    print("-" * (30 + 18 * len(templates_to_run)))

    # Common metrics to compare
    comparison_metrics = [
        "position_consistency",
        "position_bias_rate",
        "bias_toward_first_position",
        "bias_toward_second_position",
        "tie_inconsistency_rate",
    ]

    for metric_name in comparison_metrics:
        values = [all_metrics[tmpl].get(metric_name, 0) for tmpl, _ in templates_to_run]
        cells = f"{metric_name:<30}" + "".join(f"{v:<18.4f}" for v in values)
        print(cells)

    # Accuracy metrics (if available)
    if "accuracy" in all_metrics[templates_to_run[0][0]]:
        print("-" * (30 + 18 * len(templates_to_run)))
        for metric_name in ["ab_accuracy", "ba_accuracy", "overall_accuracy", "accuracy_gap"]:
            values = [
                all_metrics[tmpl].get("accuracy", {}).get(metric_name, 0)
                for tmpl, _ in templates_to_run
            ]
            cells = f"{'accuracy.'+metric_name:<30}" + "".join(f"{v:<18.4f}" for v in values)
            print(cells)

    print("=" * 80)
    logger.info("B3 evaluation complete: difficulty=%s, run=%s", args.difficulty, run_name)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
