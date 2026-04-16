"""
metrics.py
----------
Compute and display metrics from experiment results.

Key metrics per experiment
--------------------------
B1 Position Bias
  - position_consistency      : % of pairs where label is consistent across AB / BA
  - position_bias_rate        : % where AB label ≠ (flipped BA label)
  - bias_toward_first/second  : breakdown of inconsistencies by direction
  - accuracy.ab_accuracy      : % correct when better response is in position A
  - accuracy.ba_accuracy      : % correct when better response is in position B
  - accuracy.accuracy_gap     : ab_accuracy − ba_accuracy (requires gold_labels)
B2 Agreement across conditions
  - Todo better
B3 Thinking Budget – accuracy against gold
  - Todo better
B4 Agreement across templates (if applicable)
- Todo better
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
    gold_labels: Optional[dict[str, str]] = None,  # prompt_id -> "A"/"B"/"C"
) -> dict:
    """
    Measures if the model prefers responses based on their position (A or B)
    rather than their actual quality.

    Each prompt is judged twice: once in original order (AB) and once flipped (BA).
    If the model is consistent, flipping the order should flip the label
    (e.g. AB says "A" → BA should say "B", because the same response moved).

    If gold_labels is provided, also computes per-condition accuracy and the
    accuracy gap (AB accuracy - BA accuracy). A large gap means the judge is
    getting answers right in AB partly because it prefers position A, not because
    it recognises the better response.
    """
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
    flip_map = {"A": "B", "B": "A", "C": "C"}
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
            consistent_count += 1
        elif original_label == "A" and flipped_label == "A":
            always_picks_first_count += 1
        elif original_label == "B" and flipped_label == "B":
            always_picks_second_count += 1
        else:
            tie_inconsistency_count += 1

    # Step 3: Compute consistency metrics
    total_pairs = len(prompts_with_both_orders)

    metrics = {
        "n_pairs_evaluated": total_pairs,
        "position_consistency": round(consistent_count / total_pairs, 4) if total_pairs else 0,
        "position_bias_rate": round(1 - consistent_count / total_pairs, 4) if total_pairs else 0,
        "bias_toward_first_position": round(always_picks_first_count / total_pairs, 4) if total_pairs else 0,
        "bias_toward_second_position": round(always_picks_second_count / total_pairs, 4) if total_pairs else 0,
        "tie_inconsistency_rate": round(tie_inconsistency_count / total_pairs, 4) if total_pairs else 0,
    }

    # Step 4: Accuracy against gold labels (optional)
    if gold_labels is not None:
        ab_correct, ab_total = 0, 0
        ba_correct, ba_total = 0, 0

        for prompt_id, ab_label in original_order_labels.items():
            gold = gold_labels.get(prompt_id)
            if gold is None:
                continue
            ab_correct += int(ab_label == gold)
            ab_total += 1

        for prompt_id, ba_label in flipped_order_labels.items():
            gold = gold_labels.get(prompt_id)
            if gold is None:
                continue
            # Gold label is defined for the original order (better response = A).
            # In BA the responses are swapped, so the correct verdict is flipped.
            ba_correct += int(ba_label == flip_map[gold])
            ba_total += 1

        ab_acc = round(ab_correct / ab_total, 4) if ab_total else 0
        ba_acc = round(ba_correct / ba_total, 4) if ba_total else 0

        metrics["accuracy"] = {
            "ab_accuracy": ab_acc,
            "ba_accuracy": ba_acc,
            "overall_accuracy": round((ab_correct + ba_correct) / (ab_total + ba_total), 4)
                                if (ab_total + ba_total) else 0,
            "accuracy_gap": round(ab_acc - ba_acc, 4),
        }

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
