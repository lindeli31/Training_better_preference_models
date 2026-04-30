"""
Analyze B3 results (easy difficulty): compare position bias across reasoning templates.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import compute_position_bias

# Load results
results_dir = Path("results/b3_by_difficulty")
templates = {
    "expert_rater": "No Reasoning",
    "reason_then_judge": "Reason→Judge",
}

# Compute metrics for each template
all_metrics = {}
for template_id, template_label in templates.items():
    jsonl_path = results_dir / f"{template_id}_overall.jsonl"

    # Load JSONL
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line.strip()))

    # Compute metrics (no gold_labels, just position bias)
    metrics = compute_position_bias(results, gold_labels=None, exclude_ties=False)
    all_metrics[template_id] = (template_label, metrics)
    print(f"\n{template_label}:")
    print(f"  Position consistency: {metrics['position_consistency']:.4f}")
    print(f"  Position bias rate: {metrics['position_bias_rate']:.4f}")
    print(f"  Bias toward first: {metrics['bias_toward_first_position']:.4f}")
    print(f"  Bias toward second: {metrics['bias_toward_second_position']:.4f}")

# =========== Plot 1: Position Bias Metrics Comparison ===========
fig, ax = plt.subplots(figsize=(8, 5))

templates_order = ["expert_rater", "reason_then_judge"]
template_labels = [all_metrics[t][0] for t in templates_order]

metrics_names = [
    "position_consistency",
    "position_bias_rate",
    "bias_toward_first_position",
    "bias_toward_second_position",
]
metric_labels = [
    "Consistency",
    "Position Bias",
    "Primacy Bias",
    "Recency Bias",
]

x = np.arange(len(metric_labels))
width = 0.35

values_1 = [all_metrics[templates_order[0]][1][m] for m in metrics_names]
values_2 = [all_metrics[templates_order[1]][1][m] for m in metrics_names]

bars1 = ax.bar(x - width/2, values_1, width, label=template_labels[0], color="#1f407a", edgecolor="black", linewidth=0.6)
bars2 = ax.bar(x + width/2, values_2, width, label=template_labels[1], color="#e08a3c", edgecolor="black", linewidth=0.6)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_ylabel("Score", fontsize=11)
ax.set_xlabel("Metric", fontsize=11)
ax.set_title("Position Bias Metrics: Effect of Reasoning\n(Easy difficulty, n=800 pairs)", fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(results_dir / "figures" / "b3_position_bias_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: b3_position_bias_comparison.png")

# =========== Plot 2: Position Bias Components ===========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart of bias components
for idx, (template_id, (label, metrics)) in enumerate([(t, all_metrics[t]) for t in templates_order]):
    primacy = metrics["bias_toward_first_position"]
    recency = metrics["bias_toward_second_position"]

    ax1.bar([idx*2, idx*2+1], [primacy, recency], width=0.6,
            label=label, alpha=0.8, edgecolor='black', linewidth=0.6)

ax1.set_ylabel("Bias Rate", fontsize=11)
ax1.set_xlabel("Bias Direction", fontsize=11)
ax1.set_title("Position Bias Components", fontsize=11, fontweight='bold')
ax1.set_xticks([0.5, 2.5])
ax1.set_xticklabels(["Primacy Bias\n(Prefer Position A)", "Recency Bias\n(Prefer Position B)"])
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Consistency improvement
consistency_vals = [all_metrics[t][1]["position_consistency"] for t in templates_order]
colors = ["#c41e3a", "#2ecc71"]
bars = ax2.bar(template_labels, consistency_vals, color=colors, edgecolor='black', linewidth=0.6, width=0.5)

for bar, val in zip(bars, consistency_vals):
    ax2.text(bar.get_x() + bar.get_width()/2., val - 0.02,
             f'{val:.2%}', ha='center', va='top', fontsize=11, fontweight='bold', color='white')

ax2.set_ylabel("Position Consistency", fontsize=11)
ax2.set_title("Position Consistency\n(Higher is Better)", fontsize=11, fontweight='bold')
ax2.set_ylim(0.7, 0.9)
ax2.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(results_dir / "figures" / "b3_bias_components.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: b3_bias_components.png")

print("\n✓ Analysis complete. Plots saved to results/b3_by_difficulty/figures/")
