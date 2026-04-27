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
import random
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from src.inference_client import InferenceConfig, SwissAIClient
from src.dataset import DIFFICULTY_LEVELS, PairRecord, load_stratified_pairs
from src.experiments import run_evaluate_accuracy
from src.metrics import compute_accuracy_breakdown, print_summary
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
    p.add_argument("--pct-ties", type=float, default=0.0,
                   help="Fraction of the sample that should be gold_label=C "
                        "(value in [0, 1]; default 0.0 = no ties).")
    p.add_argument("--difficulties", nargs="+",
                   default=["easy", "medium", "hard"],
                   help="Difficulty buckets to include for the non-tie part "
                        "of the sample, split equally among them "
                        "(default: easy medium hard).")
    p.add_argument("--difficulty", default=None, choices=list(DIFFICULTY_LEVELS),
                   help="Shortcut for single-bucket runs: if set, overrides "
                        "--difficulties with this single bucket.")
    p.add_argument("--gold-label-seed", type=int, default=42,
                   help="Seed for the per-pair A/B re-randomisation applied at "
                        "load time. Different values give different within-bucket "
                        "gold-label distributions on the SAME prompts, which is "
                        "useful to average out gold imbalance inside a bucket.")
    return p.parse_args()


def _rerandomize_gold(pairs: list[PairRecord], seed: int) -> list[PairRecord]:
    """Flip each pair independently with p=0.5 under the given seed.

    Using PairRecord.flipped() keeps gold_label, response_a/b and extras
    mutually consistent. This re-randomises which physical response sits
    in position A without changing which one is 'better'.
    """
    rng = random.Random(seed)
    return [p.flipped() if rng.random() < 0.5 else p for p in pairs]


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

    # --difficulty (singular) wins over --difficulties (plural) when set.
    difficulties = [args.difficulty] if args.difficulty else list(args.difficulties)

    logger.info("Loading stratified dataset: split=%s | n=%d | pct_ties=%.2f | "
                "difficulties=%s | seed=%d | gold_label_seed=%d",
                args.split, args.n_pairs, args.pct_ties, difficulties,
                args.seed, args.gold_label_seed)
    pairs = load_stratified_pairs(
        split=args.split,
        n=args.n_pairs,
        pct_ties=args.pct_ties,
        seed=args.seed,
        difficulties=tuple(difficulties),
    )
    pairs = _rerandomize_gold(pairs, args.gold_label_seed)
    logger.info("Dataset ready: %d pairs (after gold re-randomisation)", len(pairs))

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
        "pct_ties": args.pct_ties,
        "difficulties": difficulties,
        "difficulty": args.difficulty,
        "gold_label_seed": args.gold_label_seed,
        "exclude_ties": args.exclude_ties,
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
