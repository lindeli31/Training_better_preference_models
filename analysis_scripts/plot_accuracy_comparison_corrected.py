import matplotlib.pyplot as plt
import numpy as np

# Corrected results with proper difficulty labels
results = {
    "easy": {"accuracy": 0.7567, "n": 1500, "color": "#28823c"},
    "medium": {"accuracy": 0.6087, "n": 1500, "color": "#e08a3c"},
    "hard": {"accuracy": 0.604, "n": 1500, "color": "#b42828"},
}

fig, ax = plt.subplots(figsize=(7, 5))

buckets = list(results.keys())
accuracies = [results[b]["accuracy"] for b in buckets]
colors = [results[b]["color"] for b in buckets]

bars = ax.bar(buckets, accuracies, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(1/3, color="gray", linestyle="--", linewidth=1.2, label="Random (1/3)")
ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
ax.set_xlabel("Difficulty bucket", fontsize=12, fontweight='bold')
ax.set_title("Judge Accuracy Across Difficulty Buckets (New thresholds)\neasy ≥1.7, medium 1.0–1.7, hard 0.3–1.0",
             fontsize=11, fontweight='bold')
ax.set_ylim(0, 0.85)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig("results/evaluate_accuracy/figures/accuracy_by_difficulty_corrected.png", dpi=150, bbox_inches='tight')
plt.close()

print("✓ Updated plot")
print("\nAccuracy by difficulty (n=1500 each):")
for b in buckets:
    print(f"  {b:8s}: {results[b]['accuracy']:.1%}")
print(f"\nDifference easy→hard: {(results['easy']['accuracy'] - results['hard']['accuracy'])*100:.2f} pp")
