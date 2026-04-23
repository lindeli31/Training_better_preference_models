"""
run_repetition_stability.py
---------------------------
B5: Repetition Stability.

For each pair, send the same (AB-order) judge call N times and measure
label agreement across repetitions.

Usage
-----
    python run_repetition_stability.py --n-pairs 100 --n-repetitions 30
    python run_repetition_stability.py --n-pairs 100 --temperature 0.7

    # Force-include specific prompts and tag the run + attach a comment:
    python run_repetition_stability.py \\
        --n-pairs 100 --pin-ids 1d7a53ffeb ce1271a1f3 \\
        --run-name pilot_with_unstable --comment "check whether the two low-agreement prompts replicate"

Each run produces two files under results/repetition_stability/:
    <base>.jsonl       # raw JudgeResponse per call
    <base>.meta.json   # args + comment + resolved prompt_ids

where <base> = <template>_<criterion>_t<temp>_<run_name_or_timestamp>.

Environment variables
---------------------
    SWISSAI_API_KEY   API key for the Swiss AI Stack
    SWISSAI_MODEL     (optional) Default model name, overridden by --model flag
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from src.inference_client import InferenceConfig, SwissAIClient
from datasets.dataset import load_dataset_pairs
from src.experiments import run_repetition_stability
from src.metrics import compute_repetition_stability, print_summary
from check_models import validate_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="B5: Repetition Stability")
    p.add_argument("--n-pairs", type=int, default=100,
                   help="Total number of dataset pairs to evaluate (pinned + sampled)")
    p.add_argument("--n-repetitions", type=int, default=30,
                   help="Number of identical repetitions per pair")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0.0 = greedy)")
    p.add_argument("--pin-ids", nargs="*", default=[],
                   help="prompt_ids that MUST be included in the run; the rest is filled randomly")
    p.add_argument("--run-name", default=None,
                   help="Short label appended to output filename (default: UTC timestamp)")
    p.add_argument("--comment", default="",
                   help="Free-text comment saved in the sidecar .meta.json")
    p.add_argument("--split", default="train",
                   help="Dataset split (train / validation)")
    p.add_argument("--criterion", default="overall")
    p.add_argument("--template", default="expert_rater")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--model", default=os.environ.get("SWISSAI_MODEL", "meta-llama/Llama-3.3-70B-Instruct"))
    p.add_argument("--base-url", default="https://api.swissai.cscs.ch/v1")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--stable-threshold", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def select_pairs(split: str, n_pairs: int, seed: int, pin_ids: list[str]):
    """
    Build the list of PairRecord to run.

    Step-by-step:
      1. Shuffle all pairs deterministically with `seed` (same shuffle every run).
      2. Resolve the pinned ids by scanning the shuffled list once and keeping
         the first occurrence of each pin (dedupe by prompt_id).
      3. Compute the complement: all pairs whose prompt_id is NOT pinned,
         in the original shuffle order (no re-shuffle).
      4. Take the first (n_pairs - n_pinned) elements of the complement to fill
         the remaining slots.
      5. Return pinned + filler, so the total is exactly n_pairs.

    Determinism guarantee:
      With the same seed, same split, and same pin_ids, this returns the same
      list every time. If the pinned ids were already in the first n_pairs of
      the unpinned seeded run, the resulting *set* of pairs is identical to
      that unpinned run (only the list order changes: pinned come first).

    Returns (pairs, pinned_found, pinned_missing).
    """
    # Step 1: deterministic shuffle of the full pool
    shuffled = load_dataset_pairs(split=split, n=None, seed=seed)
    pin_set = set(pin_ids)

    # Step 2: resolve pins (first occurrence wins, in shuffle order)
    pinned_by_id: dict = {}
    for p in shuffled:
        if p.prompt_id in pin_set and p.prompt_id not in pinned_by_id:
            pinned_by_id[p.prompt_id] = p
    pinned = list(pinned_by_id.values())
    pinned_found = list(pinned_by_id.keys())
    pinned_missing = sorted(pin_set - set(pinned_found))

    # Step 3 + 4: complement in shuffle order, take first (n_pairs - n_pinned)
    n_fill = n_pairs - len(pinned)
    if n_fill < 0:
        logger.warning("Pinned pairs (%d) exceed --n-pairs (%d); using only pinned",
                       len(pinned), n_pairs)
        filler = []
    else:
        non_pinned_in_order = [p for p in shuffled if p.prompt_id not in pin_set]
        filler = non_pinned_in_order[:n_fill]

    # Step 5: concatenate
    pairs = pinned + filler
    return pairs, pinned_found, pinned_missing


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

    logger.info("Loading dataset: split=%s | n=%d | seed=%d | pinned=%d",
                args.split, args.n_pairs, args.seed, len(args.pin_ids))
    pairs, pinned_found, pinned_missing = select_pairs(
        args.split, args.n_pairs, args.seed, args.pin_ids
    )
    logger.info("Dataset ready: %d pairs (%d pinned)", len(pairs), len(pinned_found))
    if pinned_missing:
        logger.warning("Pinned ids not found in split=%s: %s", args.split, pinned_missing)

    # Run name: user-supplied label or UTC timestamp for uniqueness
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
        "n_repetitions": args.n_repetitions,
        "temperature": args.temperature,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "pinned_ids_requested": args.pin_ids,
        "pinned_ids_found": pinned_found,
        "pinned_ids_missing": pinned_missing,
        "sampled_prompt_ids": [p.prompt_id for p in pairs if p.prompt_id not in set(pinned_found)],
    }

    t_start = time.perf_counter()
    async with SwissAIClient(config) as client:
        logger.info("=== B5: Repetition Stability (run=%s, temperature=%.2f, n_rep=%d) ===",
                    run_name, args.temperature, args.n_repetitions)
        results = await run_repetition_stability(
            client, pairs,
            n_repetitions=args.n_repetitions,
            template_id=args.template,
            criterion=args.criterion,
            output_dir=args.output_dir / "repetition_stability",
            run_name=run_name,
            meta=meta,
        )
        metrics = compute_repetition_stability(
            [r.to_dict() for r in results],
            stable_threshold=args.stable_threshold,
        )
        print_summary(f"B5: Repetition Stability  (run={run_name})", metrics)

    elapsed = time.perf_counter() - t_start
    logger.info("Experiment completed in %.1f s", elapsed)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
