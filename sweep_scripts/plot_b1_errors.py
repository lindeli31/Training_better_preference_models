"""
plot_b1_errors.py
-----------------
Two minimalist figures for the B1 position-bias sweep.

  error_breakdown.png — When the model is wrong, where do its errors go?
                        3 × 3 grid (acc-type × difficulty).  Each row is one
                        experiment; the bar shows how errors split between the
                        two wrong labels (normalised to 1.0 of error mass).
                        Accuracy is annotated at the right of every bar.

  pc_accuracy.png     — Compact overall-accuracy + position-consistency summary.
                        Bars = accuracy, diamond markers = PC.
                        One panel per difficulty, templates on x, models as colours.

Usage:
    python sweep_scripts/plot_b1_errors.py
    python sweep_scripts/plot_b1_errors.py --root results/b1_sweep_accuracy_full
    python sweep_scripts/plot_b1_errors.py --root results/b1_sweep_accuracy_full --out results/figs
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── dimensions ────────────────────────────────────────────────────────────────
MODELS_ORDER       = ["apertus", "llama33", "glm47"]
TEMPLATES_ORDER    = ["expert_rater", "llm_judge", "opro", "gepa", "opro_tree"]
BASELINE_TEMPLATES = {"expert_rater", "llm_judge"}
OPTIMIZED_TEMPLATES = {"opro", "gepa", "opro_tree"}
DIFFICULTIES       = ["easy", "medium", "hard"]
ACC_TYPES          = ["ab", "ba", "c"]

MODEL_COLORS = {"apertus": "#b42828", "llama33": "#1f407a", "glm47": "#2e8b57"}
MODEL_LABELS = {
    "apertus": "Apertus-70B",
    "llama33": "Llama-3.3-70B",
    "glm47":   "GLM-4.7-Flash",
}
TEMPLATE_LABELS = {
    "expert_rater": "expert_rater",
    "llm_judge":    "llm_judge",
    "opro":         "opro",
    "gepa":         "gepa",
    "opro_tree":    "opro_tree",
}

# Error config: per acc-type, the two possible wrong labels and their colours
# Each row uses two tones from the same hue family for a cohesive, gradient feel.
ERR_CONFIG = {
    "ab": {
        "title": "AB errors  (correct in slot A)",
        "err_keys": ["B", "C"],
        "colors":   ["#5E8CBF", "#B8D4EC"],   # deep blue → pale blue
        "leg_labels": ["wrong → B", "wrong → C"],
    },
    "ba": {
        "title": "BA errors  (correct in slot B)",
        "err_keys": ["A", "C"],
        "colors":   ["#E8835A", "#F5C4A8"],   # coral → soft peach
        "leg_labels": ["wrong → A", "wrong → C"],
    },
    "c": {
        "title": "C errors  (gold = tie)",
        "err_keys": ["A", "B"],
        "colors":   ["#6BB8A0", "#C2E0D6"],   # teal → light mint
        "leg_labels": ["wrong → A", "wrong → B"],
    },
}


# ── loading ───────────────────────────────────────────────────────────────────
def load_metrics(root: Path) -> dict:
    """Return flat index {(model, template, difficulty): record}."""
    idx = {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        with open(path) as f:
            rec = json.load(f)
        meta = rec.get("_meta", {})
        m, t, d = meta.get("model_key"), meta.get("template"), meta.get("difficulty")
        if m and t and d:
            idx[(m, t, d)] = rec
    return idx


def _v(rec, *keys):
    d = rec
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


# ── Figure 1: error breakdown ─────────────────────────────────────────────────
_XLIM_RIGHT = 1.22   # extra space to the right for the accuracy text


def plot_error_breakdown(idx: dict, out: Path) -> None:
    """
    3 rows (acc_type) × 3 cols (difficulty).
    Each cell: one horizontal bar per experiment (error-direction breakdown)
    + a compact colour-coded accuracy bar to the right.
    A dashed line separates baseline templates from optimised ones.
    """
    n_rows, n_cols = len(ACC_TYPES), len(DIFFICULTIES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(20, 3.8 * n_rows),
        gridspec_kw={"hspace": 0.45, "wspace": 0.55},
    )
    fig.suptitle("B1 — Error direction breakdown",
                 fontsize=11, fontweight="bold", y=1.02)

    bar_h = 0.55

    for row, acc_type in enumerate(ACC_TYPES):
        cfg = ERR_CONFIG[acc_type]

        for col, diff in enumerate(DIFFICULTIES):
            ax = axes[row][col]

            # Collect experiments in model × template order
            exps = []
            for m in MODELS_ORDER:
                for t in TEMPLATES_ORDER:
                    rec = idx.get((m, t, diff))
                    if rec is None:
                        continue
                    n = _v(rec, "accuracy", f"n_{acc_type}") or 0
                    if n == 0:
                        continue
                    acc      = _v(rec, "accuracy", f"{acc_type}_accuracy")
                    err_dist = _v(rec, "accuracy", f"{acc_type}_error_distribution")
                    resolved = _v(rec, "_meta", "resolved_template") or t
                    exps.append((m, t, resolved, acc, err_dist))

            if not exps:
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#aaaaaa")
                continue

            # ── draw bars ────────────────────────────────────────────────────
            ys = np.arange(len(exps))
            for i, (m, t, resolved, acc, err_dist) in enumerate(exps):
                color = MODEL_COLORS[m]

                if acc is None or (isinstance(acc, float) and np.isnan(acc)):
                    continue

                # Light background band for optimised rows
                if t in OPTIMIZED_TEMPLATES:
                    ax.barh(i, _XLIM_RIGHT, height=bar_h + 0.3,
                            left=0, color="#f5f5f5", edgecolor="none", zorder=0)

                if acc >= 1.0 or not isinstance(err_dist, dict) or not any(err_dist.values()):
                    ax.barh(i, 1.0, height=bar_h, color="#e0e0e0",
                            edgecolor="none", left=0, zorder=2)
                else:
                    left = 0.0
                    for j, ek in enumerate(cfg["err_keys"]):
                        frac = err_dist.get(ek, 0.0)
                        if frac > 1e-6:
                            ax.barh(i, frac, height=bar_h, left=left,
                                    color=cfg["colors"][j],
                                    edgecolor="white", linewidth=0.4, zorder=2)
                            if frac >= 0.12:
                                txt_color = "#333333" if j == 1 else "white"
                                ax.text(left + frac / 2, i, f"{frac:.0%}",
                                        ha="center", va="center",
                                        fontsize=6.5, color=txt_color,
                                        fontweight="bold", zorder=3)
                            left += frac

                # accuracy number to the right
                ax.text(1.04, i, f"{float(acc):.0%}",
                        va="center", fontsize=7, color=color, fontweight="bold")

            # y-axis labels
            ylabels = [
                f"{MODEL_LABELS[m].split('-')[0]}  {resolved}"
                for m, t, resolved, *_ in exps
            ]
            ax.set_yticks(ys)
            ax.set_yticklabels(ylabels, fontsize=6.5)
            ax.set_xlim(0, _XLIM_RIGHT)
            ax.set_xticks([0, 0.5, 1.0])
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", alpha=0.18, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col == 0:
                ax.set_ylabel(cfg["title"], fontsize=8, labelpad=4)
            if row == 0:
                ax.set_title(diff.capitalize(), fontsize=9, fontweight="bold")

    # ── per-row error-direction legend ────────────────────────────────────────
    for row, acc_type in enumerate(ACC_TYPES):
        cfg = ERR_CONFIG[acc_type]
        handles = [
            mpatches.Patch(fc=cfg["colors"][j], ec="#555555", linewidth=0.8,
                           label=cfg["leg_labels"][j])
            for j in range(len(cfg["err_keys"]))
        ]
        # model colour legend (once, bottom row)
        if row == len(ACC_TYPES) - 1:
            model_h = [mpatches.Patch(fc=MODEL_COLORS[m], ec="none",
                                      label=MODEL_LABELS[m])
                       for m in MODELS_ORDER]
            handles = model_h + [mpatches.Patch(fc="none", ec="none", label="")] + handles
        axes[row][2].legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.04, 0.5),
            fontsize=8,
            framealpha=0.9,
            handlelength=2.0,
            handleheight=1.4,
            labelspacing=0.8,
            borderpad=0.9,
            edgecolor="#cccccc",
        )

    out_path = out / "error_breakdown.png"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 2: accuracy + PC summary ──────────────────────────────────────────
def plot_pc_accuracy(idx: dict, out: Path) -> None:
    """
    3 panels (difficulty).  Bars = overall accuracy; diamond markers = PC.
    x groups = templates, bar colour = model.
    Optimised templates (opro / gepa / opro_tree) have a tinted background band
    and are separated from baseline templates by a dashed vertical line.
    """
    n_m    = len(MODELS_ORDER)
    bar_w  = 0.16
    grp_gap = 0.72

    centres = [ti * grp_gap for ti in range(len(TEMPLATES_ORDER))]
    base_cx = [cx for ti, (t, cx) in enumerate(zip(TEMPLATES_ORDER, centres))
               if t in BASELINE_TEMPLATES]
    opt_cx  = [cx for ti, (t, cx) in enumerate(zip(TEMPLATES_ORDER, centres))
               if t in OPTIMIZED_TEMPLATES]

    # x-span of each region
    half_g = grp_gap * 0.48
    base_span = (min(base_cx) - half_g, max(base_cx) + half_g)
    opt_span  = (min(opt_cx)  - half_g, max(opt_cx)  + half_g)
    sep_x     = (max(base_cx) + min(opt_cx)) / 2

    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    fig.suptitle("B1 — Overall accuracy  &  position consistency",
                 fontsize=11, fontweight="bold")

    for ax, diff in zip(axes, DIFFICULTIES):
        # ── background bands ─────────────────────────────────────────────────
        ax.axvspan(*base_span, color="#f7f7f7", zorder=0)
        ax.axvspan(*opt_span,  color="#eef4fb", zorder=0)
        ax.axvline(sep_x, color="#bbbbbb", linewidth=0.9,
                   linestyle="--", zorder=1)

        # ── section labels (top of each panel) ───────────────────────────────
        ax.text(np.mean(base_span), 1.04, "baseline",
                ha="center", va="bottom", fontsize=7, color="#888888",
                transform=ax.get_xaxis_transform())
        ax.text(np.mean(opt_span), 1.04, "optimized",
                ha="center", va="bottom", fontsize=7, color="#5580aa",
                transform=ax.get_xaxis_transform())

        # ── bars and PC markers ───────────────────────────────────────────────
        for ti, tmpl in enumerate(TEMPLATES_ORDER):
            cx = centres[ti]
            for mi, m in enumerate(MODELS_ORDER):
                rec = idx.get((m, tmpl, diff))
                if rec is None:
                    continue

                acc = _v(rec, "accuracy", "overall_accuracy")
                pc  = _v(rec, "position_consistency")
                x   = cx + (mi - (n_m - 1) / 2) * bar_w

                if acc is not None and not np.isnan(float(acc)):
                    ax.bar(x, acc, bar_w * 0.88,
                           color=MODEL_COLORS[m], alpha=0.80,
                           edgecolor="none", zorder=2)

                if pc is not None and not np.isnan(float(pc)):
                    ax.scatter(x, pc, s=30,
                               color=MODEL_COLORS[m],
                               marker="D", zorder=5,
                               edgecolors="white", linewidth=0.6)

        ax.set_xticks(centres)
        ax.set_xticklabels(
            [TEMPLATE_LABELS.get(t, t) for t in TEMPLATES_ORDER],
            fontsize=7.5, rotation=20, ha="right",
        )
        ax.set_ylim(0, 1.12)
        ax.axhline(1 / 3, color="gray", linewidth=0.7,
                   linestyle=":", alpha=0.5, zorder=1)
        ax.grid(axis="y", alpha=0.15, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(diff.capitalize(), fontsize=9, fontweight="bold")

    axes[0].set_ylabel("Score", fontsize=9)

    model_handles = [
        mpatches.Patch(fc=MODEL_COLORS[m], ec="none", label=MODEL_LABELS[m])
        for m in MODELS_ORDER
    ]
    type_handles = [
        mpatches.Patch(fc="#777", ec="none", alpha=0.80, label="Accuracy (bar)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#777",
                   markersize=6, label="Position consistency (◆)"),
    ]
    fig.legend(
        handles=model_handles + type_handles,
        loc="lower center", ncol=len(model_handles) + len(type_handles),
        fontsize=8, bbox_to_anchor=(0.5, -0.06), frameon=True,
    )

    out_path = out / "pc_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("results/b1_sweep"))
    p.add_argument("--out",  type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.out or (args.root / "figures")

    idx = load_metrics(args.root)
    if not idx:
        print(f"No metrics.json files found under {args.root}.")
        return

    print(f"Loaded {len(idx)} experiments.")
    plot_error_breakdown(idx, out)
    plot_pc_accuracy(idx, out)
    print(f"\nFigures saved to {out}/")


if __name__ == "__main__":
    main()
