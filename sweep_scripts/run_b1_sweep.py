"""
run_b1_sweep.py
---------------
Run B1 (position bias) for all 27 combinations of:
  - difficulty  : easy / medium / hard
  - model       : Apertus-70B, Llama-3.3-70B, GLM-4.7-Flash
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
import sys
import time
from pathlib import Path

# ensure project root is on the path when running from sweep_scripts/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

from check_models import validate_model
from src.datasets.dataset import load_dataset_pairs
from eval.experiments import run_position_bias
from core.inference_client import InferenceConfig, SwissAIClient
from eval.metrics import compute_position_bias

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = [
    "swiss-ai/Apertus-70B-Instruct-2509",
    "meta-llama/Llama-3.3-70B-Instruct",
    "zai-org/GLM-4.7-Flash",
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
    if "GLM-4.7" in name:
        return "glm47"
    return name.lower()[:20]


async def run_one(
    model: str,
    template: str,
    difficulty: str,
    api_key: str,
    concurrency: int,
    dry_run: bool,
    force: bool = False,
    output_root: Path = OUTPUT_ROOT,
) -> dict | None:
    mkey = model_key(model)
    out_dir = output_root / mkey / template / difficulty
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists() and not dry_run and not force:
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
    p.add_argument("--force", action="store_true",
                   help="Re-run even if metrics.json already exists")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max concurrent API requests per experiment")
    p.add_argument("--out", type=Path, default=OUTPUT_ROOT,
                   help="Root output directory (default: results/b1_sweep)")
    p.add_argument("--models", nargs="+", metavar="MODEL_KEY",
                   choices=[model_key(m) for m in MODELS],
                   help="Restrict to these model keys (e.g. apertus llama33 glm47)")
    p.add_argument("--templates", nargs="+", metavar="TEMPLATE",
                   choices=TEMPLATES,
                   help="Restrict to these templates (e.g. expert_rater llm_judge opro)")
    p.add_argument("--difficulties", nargs="+", metavar="DIFFICULTY",
                   choices=DIFFICULTIES,
                   help="Restrict to these difficulty levels (e.g. easy medium hard)")
    args = p.parse_args()

    model_keys_filter    = set(args.models)     if args.models      else None
    templates_filter     = set(args.templates)  if args.templates   else None
    difficulties_filter  = set(args.difficulties) if args.difficulties else None

    active_models      = [m for m in MODELS      if model_keys_filter   is None or model_key(m) in model_keys_filter]
    active_templates   = [t for t in TEMPLATES   if templates_filter    is None or t in templates_filter]
    active_difficulties = [d for d in DIFFICULTIES if difficulties_filter is None or d in difficulties_filter]

    api_key = os.environ.get("SWISSAI_API_KEY", "")
    if not api_key and not args.dry_run:
        logger.warning("SWISSAI_API_KEY not set — requests may fail")

    if not args.dry_run:
        for model in active_models:
            validate_model(model, base_url=BASE_URL, api_key=api_key)

    total = len(active_models) * len(active_templates) * len(active_difficulties)
    logger.info("Planning %d experiments (%d models × %d templates × %d difficulties)",
                total, len(active_models), len(active_templates), len(active_difficulties))

    t0 = time.perf_counter()
    done = 0
    for model in active_models:
        for template in active_templates:
            for difficulty in active_difficulties:
                done += 1
                logger.info("[%d/%d] %s | %s | %s",
                            done, total, model_key(model), template, difficulty)
                await run_one(model, template, difficulty, api_key,
                              args.concurrency, args.dry_run, args.force,
                              output_root=args.out)

    elapsed = time.perf_counter() - t0
    logger.info("Finished %d experiments in %.1f s", total, elapsed)


if __name__ == "__main__":
    asyncio.run(main())
