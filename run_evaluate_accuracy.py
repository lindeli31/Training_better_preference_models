"""
run_evaluate_accuracy.py
------------------------
Accuracy evaluation (AB-only, one call per pair).

Loads pairs from helpsteer2_{split}_full.json (needs score_a/score_b/score_gap
in extras), dispatches one AB-order judge call per pair, and saves enriched
records (JudgeResponse fields + gold_label + score_gap + is_correct) so
downstream analysis can bucket errors by score difficulty.

Usage
-----
    python run_evaluate_accuracy.py --n-pairs 500
    python run_evaluate_accuracy.py --n-pairs 500 --template expert_rater --criterion overall
    python run_evaluate_accuracy.py --n-pairs 500 --run-name baseline --comment "first full-accuracy pass"

Each run produces two files under results/evaluate_accuracy/:
    <base>.jsonl       # enriched records (one per pair)
    <base>.meta.json   # args + comment + prompt_ids

where <base> = <template>_<criterion>[_<run_name>].

Environment variables
---------------------
    SWISSAI_API_KEY   API key for the Swiss AI Stack
    SWISSAI_MODEL     (optional) Default model name, overridden by --model
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from core.inference_client import InferenceConfig, SwissAIClient
from src.datasets.dataset import load_dataset_pairs
from eval.experiments import run_evaluate_accuracy
from eval.metrics import compute_accuracy_breakdown, print_summary
from check_models import validate_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Accuracy evaluation (AB-only, enriched output)")
    p.add_argument("--n-pairs", type=int, default=500,
                   help="Number of dataset pairs to evaluate")
    p.add_argument("--split", default="train",
                   help="Dataset split (train / validation)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for dataset shuffle (determinism)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0.0 = greedy)")
    p.add_argument("--template", default="expert_rater")
    p.add_argument("--criterion", default="overall")
    p.add_argument("--run-name", default=None,
                   help="Short label appended to output filename (default: UTC timestamp)")
    p.add_argument("--comment", default="",
                   help="Free-text comment saved in the sidecar .meta.json")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--model", default=os.environ.get("SWISSAI_MODEL", "meta-llama/Llama-3.3-70B-Instruct"))
    p.add_argument("--base-url", default="https://api.swissai.cscs.ch/v1")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--exclude-ties", action="store_true",
                   help="Exclude pairs with gold_label=C from accuracy computation.")
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

    logger.info("Loading dataset (full=True): split=%s | n=%d | seed=%d",
                args.split, args.n_pairs, args.seed)
    pairs = load_dataset_pairs(
        split=args.split, n=args.n_pairs, seed=args.seed, full=True
    )
    logger.info("Dataset ready: %d pairs", len(pairs))

    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    meta = {
        "comment": args.comment,
        "run_name": run_name,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": args.model,
        "template": args.template,
        "criterion": args.criterion,
        "split": args.split,
        "n_pairs_requested": args.n_pairs,
        "n_pairs_actual": len(pairs),
        "temperature": args.temperature,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "prompt_ids": [p.prompt_id for p in pairs],
    }

    t_start = time.perf_counter()
    async with SwissAIClient(config) as client:
        logger.info("=== Accuracy evaluation (run=%s, template=%s, criterion=%s, temperature=%.2f) ===",
                    run_name, args.template, args.criterion, args.temperature)
        enriched = await run_evaluate_accuracy(
            client, pairs,
            template_id=args.template,
            criterion=args.criterion,
            output_dir=args.output_dir / "evaluate_accuracy",
            run_name=run_name,
            meta=meta,
        )
        metrics = compute_accuracy_breakdown(enriched, exclude_ties=args.exclude_ties)
        print_summary(f"Accuracy breakdown  (run={run_name})", metrics)

    elapsed = time.perf_counter() - t_start
    logger.info("Experiment completed in %.1f s", elapsed)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
