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
from src.core.inference_client import InferenceConfig, SwissAIClient
from src.dataset.dataset import load_dataset_pairs, load_stratified_pairs
from src.runall_prompt_optimization.opro_position_bias import run_opro
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
    p.add_argument("--n-train", type=int, default=800,
                   help="Total training pairs to load (matches GEPA default)")
    p.add_argument("--n-val", type=int, default=200,
                   help="Validation pairs to load (matches GEPA default)")
    p.add_argument("--stratified", action="store_true",
                   help="Use balanced mix across difficulty tiers (easy/medium/hard, ~n/3 each). "
                        "Matches GEPA's stratified setup.")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output-dir", type=Path, default=Path("results/opro"))
    p.add_argument("--model", default=os.environ.get("SWISSAI_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                   help="Judge model (evaluates prompts on pairs)")
    p.add_argument("--base-url", default="https://api.swissai.cscs.ch/v1")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--thinking", choices=["off", "on"], default="off",
                   help="enable_thinking for the judge client. "
                        "'off' (default) disables reasoning pass; 'on' enables it.")
    p.add_argument("--proposer-model", default=None,
                   help="Optional separate model for the OPRO meta-LM that "
                        "proposes new prompts. Defaults to --model (shared client).")
    p.add_argument("--proposer-thinking", choices=["off", "on", "inherit"], default="inherit",
                   help="enable_thinking for the proposer client. "
                        "'inherit' (default) reuses --thinking.")
    return p.parse_args()


async def main(args):
    api_key = os.environ.get("SWISSAI_API_KEY", "")
    validate_model(args.model, base_url=args.base_url, api_key=api_key)

    def _thinking_body(mode: str) -> dict:
        return {"chat_template_kwargs": {"enable_thinking": mode == "on"}}

    judge_config = InferenceConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        concurrent_requests=args.concurrency,
        extra_body=_thinking_body(args.thinking),
    )

    proposer_model = args.proposer_model or args.model
    proposer_thinking = args.thinking if args.proposer_thinking == "inherit" else args.proposer_thinking
    use_separate_proposer = (proposer_model != args.model) or (proposer_thinking != args.thinking)
    if use_separate_proposer:
        validate_model(proposer_model, base_url=args.base_url, api_key=api_key)
        proposer_config = InferenceConfig(
            base_url=args.base_url,
            api_key=api_key,
            model=proposer_model,
            concurrent_requests=args.concurrency,
            extra_body=_thinking_body(proposer_thinking),
        )
    logger.info("judge=%s (thinking=%s) proposer=%s (thinking=%s)",
                args.model, args.thinking, proposer_model, proposer_thinking)

    if args.stratified:
        logger.info("Loading stratified mix (easy/medium/hard, ~n/3 per tier)")
        train_pairs = load_stratified_pairs(split="train", n=args.n_train, seed=args.seed)
        val_pairs = load_stratified_pairs(split="validation", n=args.n_val, seed=args.seed)
    else:
        logger.info("Loading train pairs (n=%d) and val pairs (n=%d)", args.n_train, args.n_val)
        train_pairs = load_dataset_pairs(split="train", n=args.n_train, seed=args.seed)
        val_pairs = load_dataset_pairs(split="validation", n=args.n_val, seed=args.seed)
    logger.info("Loaded %d train, %d val pairs", len(train_pairs), len(val_pairs))

    async with SwissAIClient(judge_config) as client:
        proposer_ctx = SwissAIClient(proposer_config) if use_separate_proposer else None
        proposer_client = await proposer_ctx.__aenter__() if proposer_ctx else None
        try:
            result = await run_opro(
                client,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                n_iterations=args.n_iterations,
                eval_pairs=args.eval_pairs,
                seed=args.seed,
                output_dir=args.output_dir,
                proposer_client=proposer_client,
            )
        finally:
            if proposer_ctx:
                await proposer_ctx.__aexit__(None, None, None)

    def _fmt(m: dict) -> str:
        return (f"consistency: {m['position_consistency']:.3f}  "
                f"accuracy: {m['accuracy']:.3f}  "
                f"bias_rate: {m['position_bias_rate']:.3f}")

    print(f"\n{'='*70}")
    print(f"OPRO Prompt Optimisation Results")
    print(f"{'='*70}")
    print(f"  Baseline (train) → {_fmt(result.baseline_train)}")
    print(f"  Baseline (val)   → {_fmt(result.baseline_val)}")
    print(f"  OPRO     (train) → {_fmt(result.train)}")
    print(f"  OPRO     (val)   → {_fmt(result.val)}")
    print(f"\n  (subset GEPA-style score used during OPRO search: {result.best_train_score:.3f})")
    print(f"\nBest prompt:\n{result.best_prompt}")
    print(f"{'='*70}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
