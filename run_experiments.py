"""
run_experiments.py
------------------
Top-level orchestrator.  Run all (or selected) experiments against the
Swiss AI Stack using Qwen3-30B-A3B-Instruct-2507.

Usage
-----
    # Run everything
    python run_experiments.py

    # Run only position bias and thinking budget
    python run_experiments.py --experiments position_bias thinking_budget

    # Use fewer pairs for a quick smoke test
    python run_experiments.py --n-pairs 20 --experiments position_bias

    # Specify a different model or thinking budget
    python run_experiments.py --thinking-budget 1024

Environment
-----------
    SWISSAI_API_KEY   API key for https://serving.swissai.svc.cscs.ch/
"""

import argparse
import asyncio
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from src.inference_client import InferenceConfig, SwissAIClient
from src.dataset import load_dataset_pairs
from src.experiments import (
    run_position_bias,
    run_template_sensitivity,
    run_reasoning_depth,
    run_input_sensitivity,
    run_user_prompt_structure,
    USER_PROMPT_STRUCTURE_VARIANTS,
)
from src.metrics import (
    compute_position_bias,
    compute_pairwise_agreement,
    compute_thinking_accuracy,
    compute_stratified_metrics,
    print_summary,
    load_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALL_EXPERIMENTS = [
    "position_bias",
    "template_sensitivity",
    "reasoning_depth",
    "input_sensitivity",
    "user_prompt_structure",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM Judge Bias/Sensitivity Experiments")
    p.add_argument("--experiments", nargs="+", default=ALL_EXPERIMENTS,
                   choices=ALL_EXPERIMENTS, help="Which experiments to run")
    p.add_argument("--n-pairs", type=int, default=200,
                   help="Number of dataset pairs to evaluate")
    p.add_argument("--dataset", default="nvidia/HelpSteer2",
                   help="HuggingFace preference dataset to load")
    p.add_argument("--split", default="validation",
                   help="Dataset split (train / validation / test)")
    p.add_argument("--criterion", default="helpful",
                   help="Evaluation criterion (helpful / quality / accurate / ...)")
    p.add_argument("--template", default="expert_rater",
                   help="Judge template ID for position_bias experiment")
    p.add_argument("--output-dir", type=Path, default=Path("results"),
                   help="Root directory to save JSONL result files")
    p.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct",
                   help="Model identifier on the Swiss AI stack")
    p.add_argument("--base-url", default="https://api.swissai.cscs.ch/v1",
                   help="Swiss AI inference base URL")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max concurrent requests to the API")
    p.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling")
    p.add_argument("--stratify", action="store_true",
                   help="Sample proportionally across difficulty strata (easy/medium/hard)")
    p.add_argument("--randomize-position", action="store_true",
                   help="Randomly flip 50%% of pairs so gold_label is ~50%% A/B "
                        "(required for valid positional bias measurement)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args):
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    api_key = os.environ.get("SWISSAI_API_KEY", "")
    if not api_key:
        logger.warning(
            "SWISSAI_API_KEY not set. Requests may fail unless the endpoint is open."
        )

    config = InferenceConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        concurrent_requests=args.concurrency,
    )

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    logger.info(
        "Loading dataset: %s | split=%s | n=%d | seed=%d | stratify=%s | randomize_position=%s",
        args.dataset, args.split, args.n_pairs, args.seed,
        args.stratify, args.randomize_position,
    )
    pairs = load_dataset_pairs(
        split=args.split,
        n=args.n_pairs,
        seed=args.seed,
        stratify=args.stratify,
        randomize_position=args.randomize_position,
    )
    logger.info("Dataset ready: %d pairs", len(pairs))

    # Gold labels for accuracy computation (B3)
    gold_labels = {p.prompt_id: p.gold_label for p in pairs}

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    t_start = time.perf_counter()

    async with SwissAIClient(config) as client:

        # ---- B1: Position Bias ----------------------------------------
        if "position_bias" in args.experiments:
            logger.info("=== B1: Position Bias ===")
            pb_results = await run_position_bias(
                client, pairs,
                template_id=args.template,
                criterion=args.criterion,
                output_dir=args.output_dir / "position_bias",
            )
            pb_dicts = [r.to_dict() for r in pb_results]
            pb_metrics = compute_position_bias(pb_dicts)
            print_summary("B1: Position Bias", pb_metrics)
            if any(p.difficulty is not None for p in pairs):
                strat = compute_stratified_metrics(
                    pb_dicts, pairs, "difficulty", compute_position_bias
                )
                print_summary("B1: Position Bias — by difficulty", strat)

        # ---- B2: Template Sensitivity ----------------------------------
        if "template_sensitivity" in args.experiments:
            logger.info("=== B2: Template Sensitivity ===")
            ts_results = await run_template_sensitivity(
                client, pairs,
                criterion=args.criterion,
                output_dir=args.output_dir / "template_sensitivity",
            )
            ts_metrics = compute_pairwise_agreement(
                [r.to_dict() for r in ts_results], group_by="condition"
            )
            print_summary("B2: Template Sensitivity", ts_metrics)

        # ---- B3: Reasoning Depth ----------------------------------------
        if "reasoning_depth" in args.experiments:
            logger.info("=== B3: Reasoning Depth ===")
            rd_results = await run_reasoning_depth(
                client, pairs,
                criterion=args.criterion,
                output_dir=args.output_dir / "reasoning_depth",
            )
            rd_metrics = compute_thinking_accuracy(
                [r.to_dict() for r in rd_results], gold_labels
            )
            print_summary("B3: Reasoning Depth", rd_metrics)

        # ---- B4: Input Sensitivity ------------------------------------
        if "input_sensitivity" in args.experiments:
            logger.info("=== B4: Input Sensitivity ===")
            is_results = await run_input_sensitivity(
                client, pairs,
                output_dir=args.output_dir / "input_sensitivity",
            )
            is_metrics = compute_pairwise_agreement(
                [r.to_dict() for r in is_results], group_by="condition"
            )
            print_summary("B4: Input Sensitivity", is_metrics)

        # ---- B5: User-Prompt Structure ---------------------------------
        if "user_prompt_structure" in args.experiments:
            logger.info("=== B5: User-Prompt Structure ===")
            ups_results = await run_user_prompt_structure(
                client, pairs,
                criterion=args.criterion,
                output_dir=args.output_dir / "user_prompt_structure",
            )
            ups_dicts = [r.to_dict() for r in ups_results]

            # Per-template position consistency.
            # rsplit("_", 1) splits on the last underscore only, so template names
            # containing underscores (e.g. blind_criterion_first) are preserved.
            # New dicts are created (no in-place mutation) to avoid cross-loop corruption.
            for tmpl in USER_PROMPT_STRUCTURE_VARIANTS:
                tmpl_dicts = [
                    {**r, "condition": r["condition"].rsplit("_", 1)[1]}
                    for r in ups_dicts
                    if r["condition"].rsplit("_", 1)[0] == tmpl
                ]
                bias = compute_position_bias(tmpl_dicts)
                print_summary(f"B5: Position consistency — {tmpl}", bias)

            # Cross-condition agreement: AB-order only, template name as condition.
            ab_dicts = [
                {**r, "condition": r["condition"].rsplit("_", 1)[0]}
                for r in ups_dicts
                if r["condition"].endswith("_AB")
            ]
            ups_agreement = compute_pairwise_agreement(ab_dicts, group_by="condition")
            print_summary("B5: User-Prompt Structure — cross-condition agreement", ups_agreement)

            # Accuracy vs. gold (AB order only)
            ups_accuracy = compute_thinking_accuracy(ab_dicts, gold_labels)
            print_summary("B5: User-Prompt Structure — accuracy vs. gold", ups_accuracy)

    elapsed = time.perf_counter() - t_start
    logger.info("All experiments completed in %.1f s", elapsed)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
