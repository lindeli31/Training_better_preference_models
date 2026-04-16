"""
run_experiments.py
------------------
Top-level orchestrator for running all LLM judge bias and sensitivity
experiments. This script ties together the full pipeline:

  1. Reads command-line arguments (which experiments, how many pairs, etc.)
  2. Loads the API key and validates the model on the Swiss AI Stack
  3. Loads the dataset pairs from the local JSON files
  4. Runs the selected experiments, each of which sends judge calls to the
     LLM via the async SwissAIClient and saves results to JSONL
  5. Computes and prints summary metrics for each experiment

Available experiments:
  - position_bias        (B1): does the model prefer A or B based on position?
  - template_sensitivity (B2): does the judgment change with different templates?
  - reasoning_depth      (B3): does reasoning before judging improve accuracy?
  - input_sensitivity    (B4): does minor paraphrasing change the judgment?

Usage examples
--------------
    # Run all experiments with default settings
    python run_experiments.py

    # Run only position bias and reasoning depth
    python run_experiments.py --experiments position_bias reasoning_depth

    # Use fewer pairs for a quick test
    python run_experiments.py --n-pairs 20 --experiments position_bias

    # Use a different model
    python run_experiments.py --model meta-llama/Llama-3.3-70B-Instruct

Environment variables
---------------------
    SWISSAI_API_KEY   API key for the Swiss AI Stack (required for authentication)
    SWISSAI_MODEL     (optional) Default model name, overridden by --model flag
"""

# Standard library imports
import argparse
import asyncio
import logging
import os
import time
from dotenv import load_dotenv
from pathlib import Path
from src.inference_client import InferenceConfig, SwissAIClient
from src.dataset import load_dataset_pairs, DIFFICULTY_LEVELS
from src.experiments import (
    run_position_bias,
    run_template_sensitivity,
    run_reasoning_depth,
    run_input_sensitivity,
)
from src.metrics import (
    compute_position_bias,
    compute_pairwise_agreement,
    compute_thinking_accuracy,
    print_summary,
    load_results,
)
from check_models import validate_model
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
ALL_EXPERIMENTS = ["position_bias", "template_sensitivity", "reasoning_depth", "input_sensitivity"]

# ---------------------------------------------------------------------------
# Argument parsing
# Defines command-line arguments for running experiments. Allows the user
# to select which experiments to run, how many pairs to evaluate, which
# model and template to use, and other configuration options.
# If no arguments are provided, default values are used.
# Example usage:
#   python run_experiments.py --experiments position_bias reasoning_depth --n-pairs 100 --model meta-llama/Llama-3.3-70B-Instruct
# It parses the arguments and returns an object with the configuration that will be used in the main function.
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
    p.add_argument("--criterion", default="overall",
                   help="Evaluation criterion (overall / helpful / quality / accurate / ...)")
    p.add_argument("--template", default="expert_rater",
                   help="Judge template ID for position_bias experiment")
    p.add_argument("--output-dir", type=Path, default=Path("results"),
                   help="Root directory to save JSONL result files")
    p.add_argument("--model", default=os.environ.get("SWISSAI_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
                   help="Model identifier on the Swiss AI stack")
    p.add_argument("--base-url", default="https://api.swissai.cscs.ch/v1",
                   help="Swiss AI inference base URL")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max concurrent requests to the API")
    p.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling")
    p.add_argument("--difficulty", default=None, choices=DIFFICULTY_LEVELS,
                   help="Filter pairs by difficulty (easy / medium / hard). "
                        "Requires the full dataset variant (helpsteer2_{split}_full.json).")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main function
# This is the entry point of the experiment pipeline. It is declared as
# "async" because it uses asynchronous (non-blocking) calls to the API:
# instead of waiting for each API request to finish before sending the next,
# async allows sending many requests in parallel and collecting results as
# they arrive. This is essential for efficiency, since we make hundreds of
# API calls per experiment and running them one by one would be very slow.
# ---------------------------------------------------------------------------

async def main(args):
    # ------------------------------------------------------------------
    # Step 1: Configuration
    # Read the API key from the environment variable SWISSAI_API_KEY.
    # Then validate that the chosen model is available on the server.
    # Finally, create an InferenceConfig object with all settings
    # (URL, API key, model name, concurrency level).
    # ------------------------------------------------------------------

    api_key = os.environ.get("SWISSAI_API_KEY", "")
    if not api_key:
        logger.warning(
            "SWISSAI_API_KEY not set. Requests may fail unless the endpoint is open."
        )
    # Check that the model exists on the Swiss AI stack before starting
    validate_model(args.model, base_url=args.base_url, api_key=api_key)

    # Bundle all configuration into a single object
    config = InferenceConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        concurrent_requests=args.concurrency,
    )
    # ------------------------------------------------------------------
    # Step 2: Load dataset
    # Load the pairwise comparison records from the local JSON file.
    # Each pair contains a prompt, two responses (A and B), and a gold label.
    # Also build a dictionary of gold labels (prompt_id -> "A"/"B"/"C")
    # needed later for accuracy computation in experiment B3.
    # ------------------------------------------------------------------
    logger.info("Loading dataset: %s | split=%s | n=%d | seed=%d | difficulty=%s",
                args.dataset, args.split, args.n_pairs, args.seed, args.difficulty or "all")
    pairs = load_dataset_pairs(
        split=args.split,
        n=args.n_pairs,
        seed=args.seed,
        difficulty=args.difficulty,
    )
    logger.info("Dataset ready: %d pairs", len(pairs))
    # Map each prompt_id to its gold label
    gold_labels = {p.prompt_id: p.gold_label for p in pairs}


    # ------------------------------------------------------------------
    # Step 3: Run experiments
    # Open an async connection to the Swiss AI API and run each selected
    # experiment sequentially. Each experiment:
    #   1. Builds a list of judge calls
    #   2. Sends them in parallel via client.batch_judge()
    #   3. Saves raw results to JSONL
    #   4. Computes and prints summary metrics
    # ------------------------------------------------------------------
    t_start = time.perf_counter()

    # Open a persistent HTTP session for all experiments
    async with SwissAIClient(config) as client:

        # ---- B1: Position Bias ----------------------------------------
        # Tests if the model's judgment changes when we swap the order
        # of responses A and B. Each pair is judged twice (AB and BA).
        # Metrics: position consistency, bias toward first/second position.
        if "position_bias" in args.experiments:
            logger.info("=== B1: Position Bias ===")
            position_bias_results = await run_position_bias(
                client, pairs,
                template_id=args.template,
                criterion=args.criterion,
                output_dir=args.output_dir / "position_bias",
            )
            position_bias_metrics = compute_position_bias(
                [r.to_dict() for r in position_bias_results],
                gold_labels=gold_labels,
            )
            print_summary("B1: Position Bias", position_bias_metrics)

        # ---- B2: Template Sensitivity ----------------------------------
        # Tests if the model's judgment changes when we use different
        # system prompt templates (e.g. "expert rater" vs "LLM judge").
        # Each pair is judged once per template, same order and criterion.
        # Metrics: pairwise agreement across templates, label distribution.
        if "template_sensitivity" in args.experiments:
            logger.info("=== B2: Template Sensitivity ===")
            template_results = await run_template_sensitivity(
                client, pairs,
                criterion=args.criterion,
                output_dir=args.output_dir / "template_sensitivity",
            )
            template_metrics = compute_pairwise_agreement(
                [r.to_dict() for r in template_results], group_by="condition"
            )
            print_summary("B2: Template Sensitivity", template_metrics)

        # ---- B3: Reasoning Depth ----------------------------------------
        # Tests if asking the model to reason before judging improves
        # accuracy. Compares three conditions: no reasoning, free-form
        # reasoning, and structured reasoning with criteria.
        # Metrics: accuracy vs gold labels, agreement with baseline.
        if "reasoning_depth" in args.experiments:
            logger.info("=== B3: Reasoning Depth ===")
            reasoning_results = await run_reasoning_depth(
                client, pairs,
                criterion=args.criterion,
                output_dir=args.output_dir / "reasoning_depth",
            )
            reasoning_metrics = compute_thinking_accuracy(
                [r.to_dict() for r in reasoning_results], gold_labels
            )
            print_summary("B3: Reasoning Depth", reasoning_metrics)

        # ---- B4: Input Sensitivity ------------------------------------
        # Tests if minor paraphrasing of the same template changes the
        # model's judgment. Uses multiple variants of the expert_rater
        # template and sweeps across different evaluation criteria.
        # Metrics: pairwise agreement across variants, volatile pairs.
        if "input_sensitivity" in args.experiments:
            logger.info("=== B4: Input Sensitivity ===")
            input_sensitivity_results = await run_input_sensitivity(
                client, pairs,
                output_dir=args.output_dir / "input_sensitivity",
            )
            input_sensitivity_metrics = compute_pairwise_agreement(
                [r.to_dict() for r in input_sensitivity_results], group_by="condition"
            )
            print_summary("B4: Input Sensitivity", input_sensitivity_metrics)

    elapsed = time.perf_counter() - t_start
    logger.info("All experiments completed in %.1f s", elapsed)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
