"""
run_b1_sweep.py
---------------
Run B1 (position bias) for all 18 combinations of:
  - difficulty  : easy / medium / hard
  - model       : Apertus-70B, Llama-3.3-70B
  - template    : expert_rater / llm_judge / opro

Uses all available pairs for each difficulty level (no sub-sampling).

Results saved to:
  results/b1_sweep/<model>/<template>/<difficulty>/
    <template>_overall.jsonl   — raw judge-call records
    metrics.json               — position-bias + accuracy metrics

Re-running skips any experiment whose metrics.json already exists.

Usage:
    python sweep_scripts/run_b1_sweep.py
    python sweep_scripts/run_b1_sweep.py --dry-run        # just print the plan, no API calls
    python sweep_scripts/run_b1_sweep.py --concurrency 4  # reduce parallel requests
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from check_models import validate_model
from datasets.dataset import load_dataset_pairs
from src.experiments import run_position_bias
from core.inference_client import InferenceConfig, SwissAIClient
from src.metrics import compute_position_bias

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = [
    "swiss-ai/Apertus-70B-Instruct-2509",
    "meta-llama/Llama-3.3-70B-Instruct",
]
TEMPLATES = ["expert_rater", "llm_judge", "opro"]
DIFFICULTIES = ["easy", "medium", "hard"]
CRITERION = "overall"
BASE_URL = "https://api.swissai.cscs.ch/v1"
OUTPUT_ROOT = Path("results/b1_sweep")


def model_key(model: str) -> str:
    name = model.split("/")[-1]
    if "Apertus" in name:
        return "apertus"
    if "Llama-3.3" in name:
        return "llama33"
    return name.lower()[:20]


async def run_one(
    model: str,
    template: str,
    difficulty: str,
    api_key: str,
    concurrency: int,
    dry_run: bool,
) -> dict | None:
    mkey = model_key(model)
    out_dir = OUTPUT_ROOT / mkey / template / difficulty
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists():
        logger.info("[SKIP] %s/%s/%s — metrics.json already exists", mkey, template, difficulty)
        with open(metrics_path) as f:
            return json.load(f)

    if dry_run:
        logger.info("[DRY-RUN] Would run %s | %s | %s", mkey, template, difficulty)
        return None

    pairs = load_dataset_pairs(split="validation", n=None, difficulty=difficulty)
    gold_labels = {p.prompt_id: p.gold_label for p in pairs}
    logger.info("Loaded %d pairs (difficulty=%s)", len(pairs), difficulty)

    config = InferenceConfig(
        base_url=BASE_URL,
        api_key=api_key,
        model=model,
        concurrent_requests=concurrency,
    )

    async with SwissAIClient(config) as client:
        results = await run_position_bias(
            client, pairs,
            template_id=template,
            criterion=CRITERION,
            output_dir=out_dir,
        )

    metrics = compute_position_bias(
        [r.to_dict() for r in results],
        gold_labels=gold_labels,
    )
    metrics["_meta"] = {
        "model": model,
        "model_key": mkey,
        "template": template,
        "difficulty": difficulty,
        "n_pairs": len(pairs),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s", metrics_path)
    return metrics


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without making API calls")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max concurrent API requests per experiment")
    args = p.parse_args()

    api_key = os.environ.get("SWISSAI_API_KEY", "")
    if not api_key and not args.dry_run:
        logger.warning("SWISSAI_API_KEY not set — requests may fail")

    if not args.dry_run:
        for model in MODELS:
            validate_model(model, base_url=BASE_URL, api_key=api_key)

    total = len(MODELS) * len(TEMPLATES) * len(DIFFICULTIES)
    logger.info("Planning %d experiments (%d models × %d templates × %d difficulties)",
                total, len(MODELS), len(TEMPLATES), len(DIFFICULTIES))

    t0 = time.perf_counter()
    done = 0
    for model in MODELS:
        for template in TEMPLATES:
            for difficulty in DIFFICULTIES:
                done += 1
                logger.info("[%d/%d] %s | %s | %s",
                            done, total, model_key(model), template, difficulty)
                await run_one(model, template, difficulty, api_key,
                              args.concurrency, args.dry_run)

    elapsed = time.perf_counter() - t0
    logger.info("Finished %d experiments in %.1f s", total, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
