"""
metrics.py
----------
Compute and display metrics from experiment results.

Key metrics per experiment
--------------------------
B1 Position Bias
  - position_consistency  : % of pairs where label is consistent across AB / BA
  - position_bias_rate    : % where AB label ≠ (flipped BA label)
  - preferred_position    : overall fraction preferring "A" (positional preference)

B2 Template Sensitivity
  - pairwise_agreement    : Fleiss' kappa / pairwise % agreement across templates
  - label_distribution    : per-template A/B/C rates
  - most_volatile_pairs   : pairs with highest disagreement across templates

B3 Thinking Budget
  - accuracy_vs_gold      : % agreement with gold label per condition
  - delta_accuracy        : improvement of thinking over no-thinking
  - consistency_vs_no_reasoning : % agreement with the no-thinking condition

B4 Input Sensitivity
  - pairwise_agreement across minor wording variants
  - criterion_drift       : how much the label changes when criterion changes
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Loading
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

def compute_position_bias(results: list[dict]) -> dict:
    """
    Expects results to contain both 'AB' and 'BA' conditions for the same prompt_id.
    """
    ab_labels: dict[str, str] = {}
    ba_labels: dict[str, str] = {}

    for r in results:
        if r["label"] is None:
            continue
        pid = r["prompt_id"]
        if r["condition"] == "AB":
            ab_labels[pid] = r["label"]
        elif r["condition"] == "BA":
            ba_labels[pid] = r["label"]

    # For a consistent pair: if AB says "A", BA should say "B" (same underlying response)
    flip_map = {"A": "B", "B": "A", "C": "C"}
    both = set(ab_labels) & set(ba_labels)

    consistent = 0
    bias_toward_a = 0
    bias_toward_b = 0
    ties = 0

    for pid in both:
        ab = ab_labels[pid]
        ba = ba_labels[pid]
        expected_ba = flip_map[ab]

        if ba == expected_ba:
            consistent += 1
        else:
            # Both say "A" means model prefers whichever is listed first
            if ab == "A" and ba == "A":
                bias_toward_a += 1
            elif ab == "B" and ba == "B":
                bias_toward_b += 1
            elif ab == "C" or ba == "C":
                ties += 1

    n = len(both)
    return {
        "n_pairs_evaluated": n,
        "position_consistency": round(consistent / n, 4) if n else 0,
        "position_bias_rate": round(1 - consistent / n, 4) if n else 0,
        "bias_toward_first_position": round(bias_toward_a / n, 4) if n else 0,
        "bias_toward_second_position": round(bias_toward_b / n, 4) if n else 0,
        "tie_inconsistency_rate": round(ties / n, 4) if n else 0,
        # Overall positional preference (A-preference in AB condition)
        "pct_preferring_A_in_AB": round(
            sum(1 for v in ab_labels.values() if v == "A") / len(ab_labels), 4
        ) if ab_labels else 0,
    }


# ---------------------------------------------------------------------------
# B2 / B4: Agreement across conditions
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
# B3: Thinking Budget – accuracy against gold
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
