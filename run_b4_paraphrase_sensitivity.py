"""
run_b4_paraphrase_sensitivity.py
---------------------------------
B4 experiment: measure how much minor semantic-preserving paraphrasing of
responses changes the judgment of LLM judges.

For each pair (original and paraphrased), each judge model runs 4 conditions:
  - original_AB  : original response_a vs response_b
  - original_BA  : original response_b vs response_a (flipped)
  - para_AB      : paraphrased response_a vs response_b
  - para_BA      : paraphrased response_b vs response_a (flipped)

Models judged:
  - meta-llama/Llama-3.3-70B-Instruct   (OPRO prompt)
  - swiss-ai/Apertus-70B-Instruct-2509  (OPRO prompt)

Metrics computed per model:
  - input_sensitivity     : agreement between original_AB and para_AB
  - position_consistency  : AB vs BA consistency (original and para separately)
  - primacy_bias_rate     : tendency to pick A regardless of quality
  - recency_bias_rate     : tendency to pick B regardless of quality
  - position_bias_delta   : change in position bias after paraphrasing
  - accuracy_original     : accuracy vs gold on original responses
  - accuracy_para         : accuracy vs gold on paraphrased responses
  - accuracy_delta        : accuracy_para − accuracy_original

Output
------
  results/b4_paraphrase_sensitivity/{model_tag}.jsonl
  results/b4_paraphrase_sensitivity/{model_tag}_metrics.json

Usage
-----
    python run_b4_paraphrase_sensitivity.py
    python run_b4_paraphrase_sensitivity.py --n-pairs 100 --concurrency 8
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.core.inference_client import InferenceConfig, SwissAIClient
from src.core.templates import _build_user_prompt, SYSTEM_OPRO_LLAMA, SYSTEM_OPRO_APERTUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://api.swissai.cscs.ch/v1"

# ---------------------------------------------------------------------------
# OPRO-optimised system prompts (from results/comparison/best_prompts.md)
# ---------------------------------------------------------------------------

OPRO_PROMPTS = {
    "meta-llama/Llama-3.3-70B-Instruct": SYSTEM_OPRO_LLAMA,
    "swiss-ai/Apertus-70B-Instruct-2509": SYSTEM_OPRO_APERTUS,
}

JUDGE_MODELS = list(OPRO_PROMPTS.keys())


def load_paraphrased(path: Path) -> list[dict]:
    with open(path) as f:
        pool = json.load(f)
    pairs = [p for p in pool if p.get("response_a_para") and p.get("response_b_para")]
    failed = len(pool) - len(pairs)
    if failed:
        logger.warning("Skipped %d pairs with empty paraphrase (API failures)", failed)
    logger.info("Loaded %d pairs from %s", len(pairs), path)
    return pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_tag(model: str) -> str:
    return model.replace("/", "_").replace("-", "_").replace(".", "_").lower()


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d records → %s", len(records), path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

DIFFICULTY_LEVELS = ("tie", "hard", "medium", "easy")


def _compute_metrics(results: list[dict]) -> dict:
    """Compute all B4 metrics from enriched result records, with per-bucket breakdown."""

    def _metrics_for_subset(subset: list[dict]) -> dict:
        """Core metric computation for an arbitrary subset of result records."""
        by_cond: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in subset:
            by_cond[r["condition"]][r["prompt_id"]] = r

        orig_ab = by_cond.get("original_AB", {})
        orig_ba = by_cond.get("original_BA", {})
        para_ab = by_cond.get("para_AB", {})
        para_ba = by_cond.get("para_BA", {})

        # Input sensitivity
        shared_ids = set(orig_ab) & set(para_ab)
        agree_count = sum(
            1 for pid in shared_ids
            if orig_ab[pid]["label"] is not None
            and para_ab[pid]["label"] is not None
            and orig_ab[pid]["label"] == para_ab[pid]["label"]
        )
        valid_shared = sum(
            1 for pid in shared_ids
            if orig_ab[pid]["label"] is not None and para_ab[pid]["label"] is not None
        )
        agreement = round(agree_count / valid_shared, 4) if valid_shared else 0.0

        volatile = sorted(
            [
                {
                    "prompt_id": pid,
                    "original_label": orig_ab.get(pid, {}).get("label"),
                    "para_label":     para_ab.get(pid, {}).get("label"),
                }
                for pid in shared_ids
                if orig_ab.get(pid, {}).get("label") != para_ab.get(pid, {}).get("label")
                and orig_ab.get(pid, {}).get("label") is not None
                and para_ab.get(pid, {}).get("label") is not None
            ],
            key=lambda x: x["prompt_id"],
        )

        # Position consistency
        def _position_consistency(ab_map: dict, ba_map: dict) -> dict:
            flip = {"A": "B", "B": "A", "C": "C"}
            shared = set(ab_map) & set(ba_map)
            consistent = always_first = always_second = other = 0
            for pid in shared:
                lab_ab = ab_map[pid]["label"]
                lab_ba = ba_map[pid]["label"]
                if lab_ab is None or lab_ba is None:
                    continue
                if lab_ba == flip.get(lab_ab):
                    consistent += 1
                elif lab_ab == "A" and lab_ba == "A":
                    always_first += 1
                elif lab_ab == "B" and lab_ba == "B":
                    always_second += 1
                else:
                    other += 1
            total = consistent + always_first + always_second + other
            if total == 0:
                return {}
            return {
                "position_consistency": round(consistent    / total, 4),
                "position_bias_rate":   round(1 - consistent / total, 4),
                "primacy_bias_rate":    round(always_first  / total, 4),
                "recency_bias_rate":    round(always_second / total, 4),
                "other_rate":           round(other         / total, 4),
                "n": total,
            }

        # Accuracy
        def _accuracy(cond_map: dict, flip_gold: bool = False) -> dict:
            flip = {"A": "B", "B": "A", "C": "C"}
            correct = total = 0
            for r in cond_map.values():
                gold  = r.get("gold_label")
                label = r.get("label")
                if gold is None or label is None:
                    continue
                effective_gold = flip[gold] if flip_gold else gold
                total += 1
                if label == effective_gold:
                    correct += 1
            return {"accuracy": round(correct / total, 4) if total else 0.0, "n": total}

        # Label distribution
        def _label_dist(cond_map: dict) -> dict:
            labels = [r["label"] for r in cond_map.values() if r["label"] is not None]
            n = len(labels) or 1
            return {k: round(labels.count(k) / n, 4) for k in ("A", "B", "C")}

        pos_orig = _position_consistency(orig_ab, orig_ba)
        pos_para = _position_consistency(para_ab, para_ba)
        acc_orig = _accuracy(orig_ab)
        acc_para = _accuracy(para_ab)

        return {
            "n": valid_shared,
            "input_sensitivity": {
                "flip_rate": round(1 - agreement, 4),
                "agreement": agreement,
                "n_valid":   valid_shared,
            },
            "accuracy": {
                "original": acc_orig["accuracy"],
                "para":     acc_para["accuracy"],
                "delta":    round(acc_para["accuracy"] - acc_orig["accuracy"], 4),
            },
            "label_rates": {
                "original_AB": {**_label_dist(orig_ab), "n": len(orig_ab)},
                "original_BA": {**_label_dist(orig_ba), "n": len(orig_ba)},
                "para_AB":     {**_label_dist(para_ab), "n": len(para_ab)},
                "para_BA":     {**_label_dist(para_ba), "n": len(para_ba)},
            },
            "position_consistency": {
                "original":           pos_orig,
                "para":               pos_para,
                "position_bias_delta": round(
                    pos_para.get("position_bias_rate", 0.0)
                    - pos_orig.get("position_bias_rate", 0.0),
                    4,
                ),
            },
            "volatile_by_difficulty": {
                d: sum(
                    1 for v in volatile
                    if orig_ab.get(v["prompt_id"], {}).get("difficulty") == d
                )
                for d in DIFFICULTY_LEVELS
            },
        }

    overall = _metrics_for_subset(results)

    # Volatile pairs (top-level, for backward-compatible output)
    by_cond_top: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_cond_top[r["condition"]][r["prompt_id"]] = r
    orig_ab_top = by_cond_top.get("original_AB", {})
    para_ab_top = by_cond_top.get("para_AB", {})
    shared_top  = set(orig_ab_top) & set(para_ab_top)
    volatile_all = sorted(
        [
            {
                "prompt_id":      pid,
                "original_label": orig_ab_top[pid]["label"],
                "para_label":     para_ab_top[pid]["label"],
            }
            for pid in shared_top
            if orig_ab_top.get(pid, {}).get("label") != para_ab_top.get(pid, {}).get("label")
            and orig_ab_top.get(pid, {}).get("label") is not None
            and para_ab_top.get(pid, {}).get("label") is not None
        ],
        key=lambda x: x["prompt_id"],
    )

    # Per-difficulty buckets
    buckets = {}
    for diff in DIFFICULTY_LEVELS:
        bucket_results = [r for r in results if r.get("difficulty") == diff]
        if bucket_results:
            buckets[diff] = _metrics_for_subset(bucket_results)

    # Build output merging overall with legacy top-level shape + buckets
    s = overall["input_sensitivity"]
    return {
        "input_sensitivity": {
            "agreement_original_vs_para": s["agreement"],
            "flip_rate":           s["flip_rate"],
            "n_pairs":             s["n_valid"],
            "n_volatile_pairs":    len(volatile_all),
            "volatile_pairs_sample": volatile_all[:10],
        },
        "primacy_recency": {
            "original": {
                "primacy_rate": overall["label_rates"]["original_AB"]["A"],
                "recency_rate": overall["label_rates"]["original_AB"]["B"],
                "tie_rate":     overall["label_rates"]["original_AB"]["C"],
                "n":            overall["label_rates"]["original_AB"]["n"],
            },
            "para": {
                "primacy_rate": overall["label_rates"]["para_AB"]["A"],
                "recency_rate": overall["label_rates"]["para_AB"]["B"],
                "tie_rate":     overall["label_rates"]["para_AB"]["C"],
                "n":            overall["label_rates"]["para_AB"]["n"],
            },
        },
        "position_consistency": {
            "original": overall["position_consistency"]["original"],
            "para":     overall["position_consistency"]["para"],
            "position_bias_delta": overall["position_consistency"]["position_bias_delta"],
        },
        "accuracy": {
            "original": {"accuracy": overall["accuracy"]["original"], "n": s["n_valid"]},
            "para":     {"accuracy": overall["accuracy"]["para"],     "n": s["n_valid"]},
            "delta":    overall["accuracy"]["delta"],
        },
        "label_distribution": {
            "original_AB": {k: v for k, v in overall["label_rates"]["original_AB"].items() if k != "n"},
            "para_AB":     {k: v for k, v in overall["label_rates"]["para_AB"].items()     if k != "n"},
        },
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

async def run_for_model(
    model: str,
    pairs: list[dict],
    api_key: str,
    concurrency: int,
    output_dir: Path,
    criterion: str = "better overall",
) -> dict:
    system_prompt = OPRO_PROMPTS[model]
    config = InferenceConfig(
        base_url=BASE_URL,
        api_key=api_key,
        model=model,
        concurrent_requests=concurrency,
        temperature=0.0,
    )

    calls = []
    for pair in pairs:
        pid = pair["prompt_id"]

        # original AB
        calls.append(dict(
            system_prompt=system_prompt,
            user_prompt=_build_user_prompt(pair["prompt"], pair["response_a"], pair["response_b"], criterion),
            prompt_id=pid, experiment_id="b4_para_sensitivity", condition="original_AB",
        ))
        # original BA (flipped)
        calls.append(dict(
            system_prompt=system_prompt,
            user_prompt=_build_user_prompt(pair["prompt"], pair["response_b"], pair["response_a"], criterion),
            prompt_id=pid, experiment_id="b4_para_sensitivity", condition="original_BA",
        ))
        # para AB
        calls.append(dict(
            system_prompt=system_prompt,
            user_prompt=_build_user_prompt(pair["prompt"], pair["response_a_para"], pair["response_b_para"], criterion),
            prompt_id=pid, experiment_id="b4_para_sensitivity", condition="para_AB",
        ))
        # para BA (flipped)
        calls.append(dict(
            system_prompt=system_prompt,
            user_prompt=_build_user_prompt(pair["prompt"], pair["response_b_para"], pair["response_a_para"], criterion),
            prompt_id=pid, experiment_id="b4_para_sensitivity", condition="para_BA",
        ))

    logger.info("[%s] Dispatching %d calls (%d pairs × 4 conditions)", model, len(calls), len(pairs))
    t0 = time.perf_counter()

    async with SwissAIClient(config) as client:
        responses = await client.batch_judge(calls)

    elapsed = time.perf_counter() - t0
    logger.info("[%s] Done in %.1fs", model, elapsed)

    # Build per-pair lookup tables
    gold       = {p["prompt_id"]: p["gold_label"]      for p in pairs}
    score_gap  = {p["prompt_id"]: p.get("score_gap")   for p in pairs}
    difficulty = {p["prompt_id"]: p.get("difficulty")  for p in pairs}

    enriched = []
    for r in responses:
        d = r.to_dict()
        d["gold_label"] = gold.get(r.prompt_id)
        d["score_gap"]  = score_gap.get(r.prompt_id)
        d["difficulty"] = difficulty.get(r.prompt_id)
        enriched.append(d)

    tag = _model_tag(model)
    _save_jsonl(enriched, output_dir / f"{tag}.jsonl")

    metrics = _compute_metrics(enriched)
    metrics_path = output_dir / f"{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"model": model, "n_pairs": len(pairs), **metrics}, f, indent=2)
    logger.info("Metrics → %s", metrics_path)

    return metrics


def _print_metrics(model: str, m: dict) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {model}")
    print(f"{'=' * 65}")
    s = m["input_sensitivity"]
    print(f"  Input sensitivity  agreement={s['agreement_original_vs_para']}  "
          f"flip_rate={s['flip_rate']}  n={s['n_pairs']}")
    pc = m["position_consistency"]
    print(f"  Position bias  original={pc['original'].get('position_bias_rate', '?')}  "
          f"para={pc['para'].get('position_bias_rate', '?')}  "
          f"delta={pc['position_bias_delta']}")
    pr = m["primacy_recency"]
    print(f"  Primacy (original): {pr['original']['primacy_rate']}  "
          f"Recency: {pr['original']['recency_rate']}")
    print(f"  Primacy (para):     {pr['para']['primacy_rate']}  "
          f"Recency: {pr['para']['recency_rate']}")
    acc = m["accuracy"]
    print(f"  Accuracy  original={acc['original']['accuracy']}  "
          f"para={acc['para']['accuracy']}  delta={acc['delta']}")
    ld = m["label_distribution"]
    print(f"  Label dist original_AB: {ld['original_AB']}")
    print(f"  Label dist para_AB:     {ld['para_AB']}")


async def main(args):
    api_key = os.environ.get("SWISSAI_API_KEY", "").strip().strip('"')
    if not api_key:
        raise ValueError("SWISSAI_API_KEY not set")

    data_path = Path("data") / f"helpsteer2_{args.split}_paraphrased.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found — run run_b4_build_paraphrase_dataset.py first"
        )

    pairs = load_paraphrased(data_path)

    output_dir = Path("results/b4_paraphrase_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or JUDGE_MODELS
    all_metrics = {}

    for model in models:
        if model not in OPRO_PROMPTS:
            logger.warning("No OPRO prompt for %s — skipping", model)
            continue
        metrics = await run_for_model(
            model, pairs, api_key, args.concurrency, output_dir, args.criterion
        )
        all_metrics[model] = metrics
        _print_metrics(model, metrics)

    # Save combined summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Summary → %s", summary_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",       default="validation")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--criterion",   default="better overall")
    p.add_argument("--models",      nargs="+", default=None,
                   help="Subset of models to run (default: both Llama and Apertus)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
