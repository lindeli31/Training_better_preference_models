"""
plot_accuracy.py
----------------
Generate figures from a run_evaluate_accuracy JSONL file:

  1. accuracy_by_bucket.png       bar chart: accuracy per hard/medium/easy bucket
  2. accuracy_confusion_matrix.png heatmap:  gold x predicted, row-normalised
  3. accuracy_label_distribution.png grouped bars: gold vs predicted label share

Usage
-----
    python -m src.plot_accuracy results/evaluate_accuracy/<file>.jsonl
    python -m src.plot_accuracy results/evaluate_accuracy/<file>.jsonl --out some/other/dir

By default, figures are saved next to the run at
  <jsonl_parent>/figures/<jsonl_stem>/
so each run has its own figures folder.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import compute_accuracy_breakdown


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path, help="Enriched JSONL from run_evaluate_accuracy")
    p.add_argument("--out", type=Path, default=None,
                   help="Directory to save PNGs (default: <jsonl_parent>/figures/<jsonl_stem>/)")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def plot_accuracy_by_bucket(metrics: dict, out: Path) -> None:
    buckets = ["hard", "medium", "easy"]
    accs = [metrics["accuracy_by_gap_bucket"][b]["accuracy"] for b in buckets]
    ns = [metrics["accuracy_by_gap_bucket"][b]["n"] for b in buckets]
    overall = metrics["accuracy_overall"]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#b42828", "#e08a3c", "#28823c"]
    bars = ax.bar(buckets, accs, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="Random (1/3)")
    ax.axhline(overall, color="#1f407a", linestyle=":", linewidth=1.2,
               label=f"Overall = {overall:.3f}")

    for bar, acc, n in zip(bars, accs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.015,
                f"{acc:.3f}\nn={n}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Score-gap difficulty bucket")
    ax.set_title("Accuracy vs gold label by difficulty bucket")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_bucket.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(metrics: dict, out: Path) -> None:
    labels = ["A", "B", "C"]
    cm = metrics["confusion_matrix_gold_x_pred"]
    mat = np.array([[cm.get(g, {}).get(p, 0) for p in labels] for g in labels], dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = mat / row_sums

    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title("Confusion matrix (row-normalised by gold)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(mat[i, j])
            frac = norm[i, j]
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, f"{frac:.2f}\n({count})",
                    ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(pred | gold)")
    fig.tight_layout()
    fig.savefig(out / "accuracy_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_label_distribution(metrics: dict, out: Path) -> None:
    labels = ["A", "B", "C"]
    n_total = metrics["n_total"]
    pred = metrics["label_distribution"]
    gold_counts = {g: sum(metrics["confusion_matrix_gold_x_pred"].get(g, {}).values())
                   for g in labels}

    gold_share = [gold_counts[l] / n_total for l in labels]
    pred_share = [pred.get(l, 0) / n_total for l in labels]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - width / 2, gold_share, width, label="Gold",
                color="#1f407a", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, pred_share, width, label="Predicted",
                color="#e08a3c", edgecolor="black", linewidth=0.5)

    for bars, counts in ((b1, gold_share), (b2, pred_share)):
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of pairs")
    ax.set_xlabel("Label")
    ax.set_ylim(0, max(max(gold_share), max(pred_share)) * 1.25)
    ax.set_title("Gold vs predicted label distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_label_distribution.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out = args.out or (args.jsonl.parent / "figures" / args.jsonl.stem)
    out.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.jsonl)
    metrics = compute_accuracy_breakdown(records)

    plot_accuracy_by_bucket(metrics, out)
    plot_confusion_matrix(metrics, out)
    plot_label_distribution(metrics, out)
    print(f"Saved 3 figures to {out}/")


if __name__ == "__main__":
    main()
