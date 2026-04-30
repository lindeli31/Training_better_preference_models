"""
Plot accuracy comparison across difficulty buckets with new thresholds.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from new threshold runs
results = {
    "easy": {
        "accuracy": 0.7567,
        "n": 1500,
        "color": "#28823c",
    },
    "medium": {
        "accuracy": 0.6087,
        "n": 1500,
        "color": "#e08a3c",
    },
    "hard": {
        "accuracy": 0.5532,
        "n": 1157,
        "color": "#b42828",
    },
}

# Plot 1: Accuracy by difficulty
fig, ax = plt.subplots(figsize=(7, 5))

buckets = list(results.keys())
accuracies = [results[b]["accuracy"] for b in buckets]
colors = [results[b]["color"] for b in buckets]
ns = [results[b]["n"] for b in buckets]

bars = ax.bar(buckets, accuracies, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

# Add value labels
for bar, acc, n in zip(bars, accuracies, ns):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
            f'{acc:.1%}\nn={n}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add random baseline
ax.axhline(1/3, color="gray", linestyle="--", linewidth=1.2, label="Random (1/3)")

ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
ax.set_xlabel("Difficulty bucket", fontsize=12, fontweight='bold')
ax.set_title("Judge Accuracy Across Difficulty Buckets\n(New thresholds: easy ≥1.7, medium 1.0–1.7, hard 0.3–1.0)",
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 0.85)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig("results/evaluate_accuracy/figures/accuracy_by_difficulty_new_thresholds.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("✓ Saved: accuracy_by_difficulty_new_thresholds.png")

# Plot 2: Accuracy gap from random
fig, ax = plt.subplots(figsize=(7, 4))

gaps = [acc - 1/3 for acc in accuracies]
colors_gap = ["#2ecc71" if g > 0.1 else "#f39c12" for g in gaps]

bars = ax.bar(buckets, gaps, color=colors_gap, edgecolor="black", linewidth=0.8, width=0.6)

for bar, gap in zip(bars, gaps):
    ax.text(bar.get_x() + bar.get_width()/2, gap + 0.01,
            f'{gap:+.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Accuracy gap from random", fontsize=11, fontweight='bold')
ax.set_xlabel("Difficulty bucket", fontsize=11, fontweight='bold')
ax.set_title("Judge advantage over random baseline (1/3)", fontsize=11, fontweight='bold')
ax.set_ylim(-0.05, 0.55)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig("results/evaluate_accuracy/figures/accuracy_gap_new_thresholds.png", dpi=150, bbox_inches='tight')
plt.close(fig)

print("✓ Saved: accuracy_gap_new_thresholds.png")

print("\nSummary:")
print(f"Easy:   {results['easy']['accuracy']:.1%} accuracy (n={results['easy']['n']})")
print(f"Medium: {results['medium']['accuracy']:.1%} accuracy (n={results['medium']['n']})")
print(f"Hard:   {results['hard']['accuracy']:.1%} accuracy (n={results['hard']['n']})")
print(f"\nDifference easy→hard: {(results['easy']['accuracy'] - results['hard']['accuracy'])*100:.2f} pp")
