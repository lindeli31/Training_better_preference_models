"""
run_opro.py
-----------
Run OPRO prompt optimisation for position bias.
Optimise on training set, validate on validation set.

Usage
-----
    python run_opro.py
    python run_opro.py --n-iterations 5 --eval-pairs 20
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from src.inference_client import InferenceConfig, SwissAIClient
from datasets.dataset import load_dataset_pairs
from src.opro_position_bias import run_opro
from check_models import validate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="OPRO prompt optimisation for position bias")
    p.add_argument("--n-iterations", type=int, default=10,
                   help="Number of OPRO iterations")
    p.add_argument("--eval-pairs", type=int, default=80,
                   help="Train pairs per evaluation (subset for speed)")
    p.add_argument("--n-train", type=int, default=500,
                   help="Total training pairs to load")
    p.add_argument("--n-val", type=int, default=150,
                   help="Validation pairs to load")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output-dir", type=Path, default=Path("results/opro"))
    p.add_argument("--model", default=os.environ.get("SWISSAI_MODEL", "meta-llama/Llama-3.3-70B-Instruct"))
    p.add_argument("--base-url", default="https://api.swissai.cscs.ch/v1")
    p.add_argument("--concurrency", type=int, default=8)
    return p.parse_args()


async def main(args):
    api_key = os.environ.get("SWISSAI_API_KEY", "")
    validate_model(args.model, base_url=args.base_url, api_key=api_key)

    config = InferenceConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        concurrent_requests=args.concurrency,
    )

    logger.info("Loading train pairs (n=%d) and val pairs (n=%d)", args.n_train, args.n_val)
    train_pairs = load_dataset_pairs(split="train", n=args.n_train, seed=args.seed)
    val_pairs = load_dataset_pairs(split="validation", n=args.n_val, seed=args.seed)
    logger.info("Loaded %d train, %d val pairs", len(train_pairs), len(val_pairs))

    async with SwissAIClient(config) as client:
        result = await run_opro(
            client,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            n_iterations=args.n_iterations,
            eval_pairs=args.eval_pairs,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    print(f"\n{'='*60}")
    print(f"Train score:      {result.best_train_score:.3f}")
    print(f"Validation score: {result.val_score:.3f}")
    print(f"\nBest prompt:\n{result.best_prompt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
