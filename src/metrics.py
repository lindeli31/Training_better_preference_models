"""
metrics.py
----------
Compute and display metrics from experiment results.

Key metrics per experiment
--------------------------
B1 Position Bias
  - position_consistency  : % of pairs where label is consistent across AB / BA
  - position_bias_rate    : % where AB label ≠ (flipped BA label)
B2 Agreement across conditions
  - Todo better
B3 Thinking Budget – accuracy against gold
  - Todo better
B4 Agreement across templates (if applicable)
- Todo better
B5 Repetition Stability
  - mean_modal_agreement : avg of count(modal label) / n_rep across prompts
  - stable_prompt_rate   : fraction of prompts with modal_agreement ≥ threshold
  - overall_label_distribution : A/B/C share across all repetitions
  - least_stable_prompts : N prompts with the lowest modal_agreement
Accuracy breakdown
  - accuracy_overall    : % of predictions matching gold_label
  - accuracy_by_gap_bucket : accuracy within hard / medium / easy buckets
  - confusion_matrix_gold_x_pred : gold → predicted label counts
  - label_distribution   : predicted label counts
"""

# Standard library imports
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np
# ---------------------------------------------------------------------------
# Loading. This funnction loads the results from a JSONL file, where each line 
# is a JSON object representing a single judgment. The expected format of each line is:
# {
#   "prompt_id": str,
#   "condition": str,  # e.g., "AB", "BA", "template1", "template2", "no_reasoning", etc.
#   "label": str,  # "A", "B", "C", or None if no valid label was extracted
#   ... (other metadata fields)
# }
# ---------------------------------------------------------------------------
def load_results(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

# ---------------------------------------------------------------------------
# B1: Position Bias
# ---------------------------------------------------------------------------

def compute_position_bias(
    results: list[dict],
    gold_labels: Optional[dict[str, str]] = None,
) -> dict:
    """
    Measures if the model prefers responses based on their position (A or B)
    rather than their actual quality.

    Each prompt is judged twice: once in original order (AB) and once flipped (BA).
    If the model is consistent, flipping the order should flip the label
    (e.g. AB says "A" → BA should say "B", because the same response moved).

    If `gold_labels` (prompt_id -> gold label in original order) is provided,
    also computes accuracy_AB and accuracy_BA: the fraction of judgments in
    each condition that match the ground truth. In BA the ground truth is the
    flipped gold label, since response_a and response_b are physically swapped.
    """
    flip_map = {"A": "B", "B": "A", "C": "C"}

    # Step 1: Separate results into original order (AB) and flipped order (BA)
    original_order_labels = {}   # prompt_id -> label when shown in AB order
    flipped_order_labels = {}    # prompt_id -> label when shown in BA order

    for result in results:
        if result["label"] is None:
            continue
        prompt_id = result["prompt_id"]
        if result["condition"] == "AB":
            original_order_labels[prompt_id] = result["label"]
        elif result["condition"] == "BA":
            flipped_order_labels[prompt_id] = result["label"]

    # Step 2: For each prompt judged in both orders, check consistency
    prompts_with_both_orders = set(original_order_labels) & set(flipped_order_labels)

    consistent_count = 0
    always_picks_first_count = 0    # model always picks position A
    always_picks_second_count = 0   # model always picks position B
    tie_inconsistency_count = 0

    for prompt_id in prompts_with_both_orders:
        original_label = original_order_labels[prompt_id]
        flipped_label = flipped_order_labels[prompt_id]
        expected_flipped_label = flip_map[original_label]

        if flipped_label == expected_flipped_label:
            # Model is consistent: picked the same response regardless of position
            consistent_count += 1
        elif original_label == "A" and flipped_label == "A":
            # Both times picked first position → bias toward position A
            always_picks_first_count += 1
        elif original_label == "B" and flipped_label == "B":
            # Both times picked second position → bias toward position B
            always_picks_second_count += 1
        else:
            # One of them is a tie → inconsistency involving ties
            tie_inconsistency_count += 1

    # Step 3: Compute metrics
    total_pairs = len(prompts_with_both_orders)

    metrics = {
        "n_pairs_evaluated": total_pairs,
        "position_consistency": round(consistent_count / total_pairs, 4) if total_pairs else 0,
        "position_bias_rate": round(1 - consistent_count / total_pairs, 4) if total_pairs else 0,
        "bias_toward_first_position": round(always_picks_first_count / total_pairs, 4) if total_pairs else 0,
        "bias_toward_second_position": round(always_picks_second_count / total_pairs, 4) if total_pairs else 0,
        "tie_inconsistency_rate": round(tie_inconsistency_count / total_pairs, 4) if total_pairs else 0,
    }

    # Step 4 (optional): accuracy vs gold per condition
    # AB: model's label should equal gold_label
    # BA: model's label should equal flip(gold_label) (responses are swapped)
    if gold_labels is not None:
        ab_correct = ab_total = 0
        for pid, label in original_order_labels.items():
            gold = gold_labels.get(pid)
            if gold is None:
                continue
            ab_total += 1
            if label == gold:
                ab_correct += 1

        ba_correct = ba_total = 0
        for pid, label in flipped_order_labels.items():
            gold = gold_labels.get(pid)
            if gold is None:
                continue
            ba_total += 1
            if label == flip_map[gold]:
                ba_correct += 1

        metrics["accuracy_AB"] = round(ab_correct / ab_total, 4) if ab_total else 0
        metrics["accuracy_BA"] = round(ba_correct / ba_total, 4) if ba_total else 0
        metrics["n_AB_evaluated"] = ab_total
        metrics["n_BA_evaluated"] = ba_total

    return metrics



# TODO:....

# ---------------------------------------------------------------------------
# B2 / B4: Agreement across conditions. To do better.
# ---------------------------------------------------------------------------

def compute_pairwise_agreement(
    results: list[dict],
    group_by: str = "condition",        # "condition" or "template_id"
) -> dict:
    """
    For each pair, collect the label from each condition/template.
    Compute:
      - overall pairwise agreement %
      - per-pair agreement rate (identify volatile pairs)
      - label distribution per condition
    """
    # pivot: prompt_id -> condition -> label
    pivot: dict[str, dict[str, str]] = defaultdict(dict)
    conditions: set[str] = set()

    for r in results:
        if r["label"] is None:
            continue
        pivot[r["prompt_id"]][r[group_by]] = r["label"]
        conditions.add(r[group_by])

    conditions = sorted(conditions)
    pair_agreements = []

    for pid, labels in pivot.items():
        cond_labels = [labels[c] for c in conditions if c in labels]
        if len(cond_labels) < 2:
            continue
        # Agreement = fraction of pairs of conditions that agree
        agrees = sum(
            1 for i in range(len(cond_labels))
            for j in range(i + 1, len(cond_labels))
            if cond_labels[i] == cond_labels[j]
        )
        total_pairs = len(cond_labels) * (len(cond_labels) - 1) // 2
        pair_agreements.append((pid, agrees / total_pairs))

    # Label distribution per condition
    label_dist: dict[str, dict[str, int]] = defaultdict(lambda: {"A": 0, "B": 0, "C": 0})
    for r in results:
        if r["label"] in ("A", "B", "C"):
            label_dist[r[group_by]][r["label"]] += 1

    volatile_pairs = sorted(pair_agreements, key=lambda x: x[1])[:10]  # lowest agreement
    overall_agreement = np.mean([a for _, a in pair_agreements]) if pair_agreements else 0

    return {
        "n_conditions": len(conditions),
        "conditions": conditions,
        "overall_pairwise_agreement": round(float(overall_agreement), 4),
        "label_distribution_per_condition": dict(label_dist),
        "most_volatile_pairs": [{"prompt_id": p, "agreement": round(a, 4)}
                                  for p, a in volatile_pairs],
    }


# ---------------------------------------------------------------------------
# B3: Thinking Budget – accuracy against gold. To do better
# ---------------------------------------------------------------------------

def compute_thinking_accuracy(
    results: list[dict],
    gold_labels: dict[str, str],     # prompt_id -> gold label
) -> dict:
    """
    For each condition (thinking level), compute accuracy vs. gold labels.
    Also compute agreement rate with the no_reasoning baseline.
    """
    # pivot: condition -> prompt_id -> label
    pivot: dict[str, dict[str, str]] = defaultdict(dict)
    for r in results:
        if r["label"] is not None:
            pivot[r["condition"]][r["prompt_id"]] = r["label"]

    accuracy: dict[str, float] = {}
    for cond, labels in pivot.items():
        correct = sum(1 for pid, lbl in labels.items()
                      if gold_labels.get(pid) == lbl)
        accuracy[cond] = round(correct / len(labels), 4) if labels else 0

    # Agreement with no_reasoning baseline
    baseline = pivot.get("no_reasoning", {})
    agreement_vs_baseline: dict[str, float] = {}
    for cond, labels in pivot.items():
        if cond == "no_reasoning":
            continue
        shared = set(labels) & set(baseline)
        if shared:
            agree = sum(1 for pid in shared if labels[pid] == baseline[pid])
            agreement_vs_baseline[cond] = round(agree / len(shared), 4)

    # Average latency per condition
    latency: dict[str, list[float]] = defaultdict(list)
    for r in results:
        latency[r["condition"]].append(r.get("latency_s", 0))

    avg_latency = {c: round(float(np.mean(v)), 3) for c, v in latency.items()}

    return {
        "accuracy_vs_gold": accuracy,
        "agreement_vs_no_reasoning": agreement_vs_baseline,
        "avg_latency_s": avg_latency,
    }


# ---------------------------------------------------------------------------
# B5: Repetition Stability
# ---------------------------------------------------------------------------

def compute_repetition_stability(
    results: list[dict],
    stable_threshold: float = 0.9,
    n_least_stable: int = 10,
) -> dict:
    """
    Group judgments by prompt_id, measure how often the model returns the same
    label across repeated identical calls.

    For each prompt:
      - modal_label      : most frequent label among its repetitions
      - modal_agreement  : count(modal_label) / n_valid_repetitions
    Aggregated:
      - mean_modal_agreement
      - stable_prompt_rate         (% of prompts with modal_agreement ≥ threshold)
      - overall_label_distribution (A/B/C share across all repetitions)
      - parse_fail_rate
      - least_stable_prompts       (bottom N by modal_agreement)
    """
    # prompt_id -> list of labels (None kept separately)
    per_prompt_labels: dict[str, list[str]] = defaultdict(list)
    per_prompt_none: dict[str, int] = defaultdict(int)
    total_calls = 0

    for r in results:
        total_calls += 1
        if r["label"] is None:
            per_prompt_none[r["prompt_id"]] += 1
        else:
            per_prompt_labels[r["prompt_id"]].append(r["label"])

    prompt_stats = []
    overall_counts = {"A": 0, "B": 0, "C": 0}
    for pid, labels in per_prompt_labels.items():
        counts = {"A": 0, "B": 0, "C": 0}
        for lbl in labels:
            if lbl in counts:
                counts[lbl] += 1
        for k in counts:
            overall_counts[k] += counts[k]
        modal_label = max(counts, key=counts.get)
        modal_count = counts[modal_label]
        modal_agreement = modal_count / len(labels)
        prompt_stats.append({
            "prompt_id": pid,
            "n_valid": len(labels),
            "n_failed": per_prompt_none.get(pid, 0),
            "modal_label": modal_label,
            "modal_agreement": round(modal_agreement, 4),
            "counts": counts,
        })

    if not prompt_stats:
        return {"error": "no prompts with valid labels"}

    agreements = [p["modal_agreement"] for p in prompt_stats]
    mean_agreement = float(np.mean(agreements))
    stable_rate = float(np.mean([a >= stable_threshold for a in agreements]))
    total_valid = sum(overall_counts.values())
    dist = {k: round(v / total_valid, 4) for k, v in overall_counts.items()} if total_valid else overall_counts
    parse_fail_rate = round(sum(per_prompt_none.values()) / total_calls, 4) if total_calls else 0

    least_stable = sorted(prompt_stats, key=lambda p: p["modal_agreement"])[:n_least_stable]

    return {
        "n_prompts": len(prompt_stats),
        "n_calls_total": total_calls,
        "mean_modal_agreement": round(mean_agreement, 4),
        "stable_prompt_rate": round(stable_rate, 4),
        "stable_threshold": stable_threshold,
        "overall_label_distribution": dist,
        "parse_fail_rate": parse_fail_rate,
        "least_stable_prompts": least_stable,
    }


# ---------------------------------------------------------------------------
# Accuracy breakdown (for run_evaluate_accuracy)
# Uses the same "difficulty" buckets as dataset.py: hard (gap ≤ 0.33),
# medium (0.33 < gap ≤ 1.0), easy (gap > 1.0). Score scale is 0-4.
# ---------------------------------------------------------------------------

def compute_accuracy_breakdown(results: list[dict]) -> dict:
    """
    Expects enriched records with at least: label, gold_label, score_gap.
    """
    buckets = [(0.33, "hard"), (1.0, "medium"), (float("inf"), "easy")]
    bucket_stats = {name: {"correct": 0, "total": 0} for _, name in buckets}
    confusion: dict = defaultdict(lambda: {"A": 0, "B": 0, "C": 0, "None": 0})
    label_dist = {"A": 0, "B": 0, "C": 0, "None": 0}

    correct = 0
    total = 0
    parse_fail = 0

    for r in results:
        total += 1
        lbl = r.get("label")
        gold = r.get("gold_label")
        gap = r.get("score_gap") if r.get("score_gap") is not None else 0.0

        key_lbl = lbl if lbl is not None else "None"
        label_dist[key_lbl] += 1
        confusion[gold][key_lbl] += 1

        if lbl is None:
            parse_fail += 1
            continue

        for upper, name in buckets:
            if gap <= upper:
                bucket_stats[name]["total"] += 1
                if lbl == gold:
                    bucket_stats[name]["correct"] += 1
                break

        if lbl == gold:
            correct += 1

    accuracy_by_bucket = {
        name: {
            "n": s["total"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] else 0.0,
        }
        for name, s in bucket_stats.items()
    }

    return {
        "n_total": total,
        "accuracy_overall": round(correct / total, 4) if total else 0.0,
        "accuracy_by_gap_bucket": accuracy_by_bucket,
        "label_distribution": label_dist,
        "confusion_matrix_gold_x_pred": {k: dict(v) for k, v in confusion.items()},
        "parse_fail_rate": round(parse_fail / total, 4) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_summary(title: str, metrics: dict, indent: int = 2) -> None:
    pad = " " * indent
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    def _print(d, level=0):
        for k, v in d.items():
            prefix = pad * (level + 1)
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                _print(v, level + 1)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"{prefix}{k}:")
                for item in v:
                    print(f"{prefix}  {item}")
            else:
                print(f"{prefix}{k}: {v}")

    _print(metrics)
