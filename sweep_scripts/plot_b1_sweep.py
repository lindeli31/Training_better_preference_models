"""
plot_b1_sweep.py
----------------
Load all metrics.json files from results/b1_sweep/ and produce comparison plots.

Expects the directory layout written by run_b1_sweep.py:
  results/b1_sweep/<model>/<template>/<difficulty>/metrics.json

Figures saved to results/b1_sweep/figures/:
  1. accuracy_overview.png      — overall/ab/ba accuracy and accuracy gap
  2. position_bias_overview.png — consistency and directional bias rates
  3. summary_heatmap.png        — 18-experiment heatmap (overall accuracy)

Usage:
    python sweep_scripts/plot_b1_sweep.py
    python sweep_scripts/plot_b1_sweep.py --root results/b1_sweep --out results/b1_sweep/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour / style constants ────────────────────────────────────────────────
MODEL_COLORS = {
    "apertus": "#b42828",
    "llama33":  "#1f407a",
    "glm47":   "#2e8b57",
}
MODEL_LABELS = {
    "apertus": "Apertus-70B",
    "llama33":  "Llama-3.3-70B",
    "glm47":   "GLM-4.7-Flash",
}
TEMPLATE_HATCHES = {
    "expert_rater": "",
    "llm_judge":    "//",
    "opro":         "xx",
}
TEMPLATE_LABELS = {
    "expert_rater": "expert_rater",
    "llm_judge":    "llm_judge",
    "opro":         "opro",
}
DIFFICULTIES = ["easy", "medium"]
MODELS_ORDER = ["apertus", "llama33", "glm47"]
TEMPLATES_ORDER = ["expert_rater", "llm_judge", "opro"]


# ── loading ──────────────────────────────────────────────────────────────────
def load_all_metrics(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.glob("*/*/*/metrics.json")):
        with open(path) as f:
            records.append(json.load(f))
    return records


def get(rec: dict, *keys, default=float("nan")):
    """Nested dict access with a fallback."""
    d = rec
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


# ── helpers for grouped bar charts ───────────────────────────────────────────
def _bar_positions(n_groups: int, n_bars: int, width: float = 0.15, gap: float = 0.55):
    """Return (group_centres, bar_offsets). Groups are spaced by `gap`."""
    centres = np.arange(n_groups) * gap
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * width
    return centres, offsets


def _grouped_bars(ax, data_matrix, group_labels, bar_labels, colors, hatches,
                  width=0.15, gap=0.55, alpha=0.85):
    """
    data_matrix : shape (n_groups, n_bars)
    Returns list of bar containers (one per bar column).
    """
    n_groups, n_bars = np.array(data_matrix).shape
    centres, offsets = _bar_positions(n_groups, n_bars, width, gap)
    containers = []
    for b in range(n_bars):
        vals = [data_matrix[g][b] for g in range(n_groups)]
        bars = ax.bar(
            centres + offsets[b], vals, width,
            color=colors[b], hatch=hatches[b],
            edgecolor="black", linewidth=0.5, alpha=alpha,
            label=bar_labels[b],
        )
        containers.append(bars)
    ax.set_xticks(centres)
    ax.set_xticklabels(group_labels, fontsize=8)
    return containers


# ── Figure 1: accuracy metrics ───────────────────────────────────────────────
def plot_accuracy_overview(records: list[dict], out: Path) -> None:
    """
    3×3 grid: rows = metric, cols = difficulty
    Each cell: x = template, bars = model (2 bars per template group)
    """
    metrics_info = [
        ("accuracy", "overall_accuracy", "Overall accuracy", (0.0, 1.05)),
        ("accuracy", "ab_accuracy",      "AB accuracy",      (0.0, 1.05)),
        ("accuracy", "ba_accuracy",      "BA accuracy",      (0.0, 1.05)),
        ("accuracy", "accuracy_gap",     "Accuracy gap (AB − BA)", (-0.5, 0.5)),
    ]

    # index records for quick lookup: (model, template, difficulty)
    idx = {
        (r["_meta"]["model_key"], r["_meta"]["template"], r["_meta"]["difficulty"]): r
        for r in records
    }

    n_rows, n_cols = len(metrics_info), len(DIFFICULTIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows),
                             sharey="row", sharex="col")
    fig.suptitle("B1 Position Bias — Accuracy Metrics", fontsize=13, fontweight="bold", y=1.01)

    bar_labels = []
    colors = []
    hatches = []
    for m in MODELS_ORDER:
        for t in TEMPLATES_ORDER:
            bar_labels.append(f"{MODEL_LABELS[m]}\n{TEMPLATE_LABELS[t]}")
            colors.append(MODEL_COLORS[m])
            hatches.append(TEMPLATE_HATCHES[t])

    for row, (top_key, sub_key, ylabel, ylim) in enumerate(metrics_info):
        for col, diff in enumerate(DIFFICULTIES):
            ax = axes[row][col]
            data = []
            for t in TEMPLATES_ORDER:
                row_vals = []
                for m in MODELS_ORDER:
                    rec = idx.get((m, t, diff))
                    if rec is None:
                        row_vals.append(float("nan"))
                    else:
                        row_vals.append(get(rec, top_key, sub_key))
                data.append(row_vals)

            # data shape: (n_templates=3, n_models=2) — flatten to 6 bars
            flat_data = [[v for row_v in data for v in row_v]]   # 1 group, 6 bars
            flat_colors  = [MODEL_COLORS[m] for _ in TEMPLATES_ORDER for m in MODELS_ORDER]
            flat_hatches = [TEMPLATE_HATCHES[t] for t in TEMPLATES_ORDER for _ in MODELS_ORDER]
            flat_labels  = [f"{MODEL_LABELS[m]}/{t}"
                            for t in TEMPLATES_ORDER for m in MODELS_ORDER]

            # Use TEMPLATES as x groups, MODELS as bars within each group
            group_data = []
            for t in TEMPLATES_ORDER:
                group_row = [get(idx.get((m, t, diff)), top_key, sub_key)
                             for m in MODELS_ORDER]
                group_data.append(group_row)

            bar_cols = [MODEL_COLORS[m] for m in MODELS_ORDER]
            bar_hatch = ["", "//", "xx"]  # one per model
            _grouped_bars(ax, group_data, TEMPLATES_ORDER, MODELS_ORDER,
                          bar_cols, bar_hatch, width=0.15, gap=0.65)

            if "gap" in sub_key:
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            else:
                ax.axhline(1 / 3, color="gray", linewidth=0.8, linestyle=":",
                           label="Random (1/3)")

            ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelsize=7)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 0:
                ax.set_title(f"Difficulty: {diff}", fontsize=9, fontweight="bold")

    # shared legend
    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABELS[m],
                       edgecolor="black")
        for m in MODELS_ORDER
    ] + [
        mpatches.Patch(facecolor="white", hatch=TEMPLATE_HATCHES[t],
                       label=TEMPLATE_LABELS[t], edgecolor="black")
        for t in TEMPLATES_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.tight_layout()
    out_path = out / "accuracy_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 2: position bias metrics ─────────────────────────────────────────
def plot_position_bias_overview(records: list[dict], out: Path) -> None:
    """
    3×3 grid: rows = metric, cols = difficulty
    """
    metrics_info = [
        ("position_consistency",       "Position consistency",       (0.0, 1.05)),
        ("position_bias_rate",         "Position bias rate",         (0.0, 1.05)),
        ("bias_toward_first_position", "Bias toward first (A)",      (0.0, 1.05)),
        ("bias_toward_second_position","Bias toward second (B)",     (0.0, 1.05)),
    ]

    idx = {
        (r["_meta"]["model_key"], r["_meta"]["template"], r["_meta"]["difficulty"]): r
        for r in records
    }

    n_rows, n_cols = len(metrics_info), len(DIFFICULTIES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows),
                             sharey="row", sharex="col")
    fig.suptitle("B1 Position Bias — Consistency & Bias Metrics",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, (key, ylabel, ylim) in enumerate(metrics_info):
        for col, diff in enumerate(DIFFICULTIES):
            ax = axes[row][col]

            group_data = []
            for t in TEMPLATES_ORDER:
                group_row = [get(idx.get((m, t, diff)), key)
                             for m in MODELS_ORDER]
                group_data.append(group_row)

            bar_cols = [MODEL_COLORS[m] for m in MODELS_ORDER]
            _grouped_bars(ax, group_data, TEMPLATES_ORDER, MODELS_ORDER,
                          bar_cols, ["", "//", "xx"], width=0.15, gap=0.65)

            ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelsize=7)

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 0:
                ax.set_title(f"Difficulty: {diff}", fontsize=9, fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABELS[m],
                       edgecolor="black")
        for m in MODELS_ORDER
    ] + [
        mpatches.Patch(facecolor="white", hatch=TEMPLATE_HATCHES[t],
                       label=TEMPLATE_LABELS[t], edgecolor="black")
        for t in TEMPLATES_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.tight_layout()
    out_path = out / "position_bias_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 3: summary heatmaps ───────────────────────────────────────────────
def plot_summary_heatmaps(records: list[dict], out: Path) -> None:
    """
    Two side-by-side heatmaps:
      left  — overall_accuracy
      right — accuracy_gap
    Rows: model × template (6 rows), Cols: difficulty (3 cols)
    """
    idx = {
        (r["_meta"]["model_key"], r["_meta"]["template"], r["_meta"]["difficulty"]): r
        for r in records
    }

    row_labels = [f"{MODEL_LABELS[m]} / {t}"
                  for m in MODELS_ORDER for t in TEMPLATES_ORDER]
    row_keys   = [(m, t) for m in MODELS_ORDER for t in TEMPLATES_ORDER]

    metrics_to_plot = [
        ("accuracy", "overall_accuracy", "Overall accuracy",   "Blues",   0.0, 1.0),
        ("accuracy", "accuracy_gap",     "Accuracy gap (AB−BA)", "RdBu_r", -0.4, 0.4),
        ("position_consistency", None,   "Position consistency","Greens",  0.0, 1.0),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                             figsize=(5 * len(metrics_to_plot), 5.5))
    fig.suptitle("B1 Summary — 27 Experiments", fontsize=12, fontweight="bold")

    for ax, (top_key, sub_key, title, cmap, vmin, vmax) in zip(axes, metrics_to_plot):
        mat = np.full((len(row_keys), len(DIFFICULTIES)), float("nan"))
        for r, (m, t) in enumerate(row_keys):
            for c, diff in enumerate(DIFFICULTIES):
                rec = idx.get((m, t, diff))
                if rec is None:
                    continue
                if sub_key is None:
                    mat[r, c] = get(rec, top_key)
                else:
                    mat[r, c] = get(rec, top_key, sub_key)

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(DIFFICULTIES)))
        ax.set_xticklabels(DIFFICULTIES, fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")

        for r in range(len(row_keys)):
            for c in range(len(DIFFICULTIES)):
                v = mat[r, c]
                if not np.isnan(v):
                    text_color = "white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.3 else "black"
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=7.5, color=text_color)

    fig.tight_layout()
    out_path = out / "summary_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 4: model comparison ───────────────────────────────────────────────
def plot_model_comparison(records: list[dict], out: Path) -> None:
    """
    Radar-style summary: for each metric, show mean across templates,
    grouped by model and faceted by difficulty.
    """
    metrics_info = [
        ("accuracy", "overall_accuracy", "Overall\naccuracy"),
        ("accuracy", "accuracy_gap",     "Accuracy\ngap"),
        ("position_consistency",  None,  "Position\nconsistency"),
        ("bias_toward_first_position", None, "Bias\ntoward A"),
        ("bias_toward_second_position", None, "Bias\ntoward B"),
    ]

    idx = {
        (r["_meta"]["model_key"], r["_meta"]["template"], r["_meta"]["difficulty"]): r
        for r in records
    }

    fig, axes = plt.subplots(1, len(DIFFICULTIES), figsize=(14, 4), sharey=False)
    fig.suptitle("B1 — Model comparison (mean across templates)",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(metrics_info))
    width = 0.3

    for ax, diff in zip(axes, DIFFICULTIES):
        for i, m in enumerate(MODELS_ORDER):
            means = []
            for top_key, sub_key, _ in metrics_info:
                vals = []
                for t in TEMPLATES_ORDER:
                    rec = idx.get((m, t, diff))
                    if rec is None:
                        continue
                    v = get(rec, top_key) if sub_key is None else get(rec, top_key, sub_key)
                    if not np.isnan(v):
                        vals.append(v)
                means.append(np.mean(vals) if vals else float("nan"))

            offset = (i - (len(MODELS_ORDER) - 1) / 2) * width
            bars = ax.bar(x + offset, means, width,
                          color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                          edgecolor="black", linewidth=0.5, alpha=0.85)
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=6.5)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(f"Difficulty: {diff}", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m[2] for m in metrics_info], fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    handles = [mpatches.Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABELS[m],
                               edgecolor="black") for m in MODELS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(MODELS_ORDER), fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    out_path = out / "model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/b1_sweep"))
    p.add_argument("--out",  type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out or (args.root / "figures")
    out.mkdir(parents=True, exist_ok=True)

    records = load_all_metrics(args.root)
    if not records:
        print(f"No metrics.json files found under {args.root}. "
              "Run run_b1_sweep.py first.")
        return

    found = {(r["_meta"]["model_key"], r["_meta"]["template"], r["_meta"]["difficulty"])
             for r in records}
    print(f"Loaded {len(records)} experiment results:")
    for m in MODELS_ORDER:
        for t in TEMPLATES_ORDER:
            for d in DIFFICULTIES:
                status = "OK" if (m, t, d) in found else "MISSING"
                print(f"  [{status}] {m} / {t} / {d}")

    plot_accuracy_overview(records, out)
    plot_position_bias_overview(records, out)
    plot_summary_heatmaps(records, out)
    plot_model_comparison(records, out)
    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
