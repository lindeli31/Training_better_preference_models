"""
make_sweep_diagram.py
---------------------
Generate a slide-ready overview of the B1 sweep setup.

Left panel  — sweep matrix  (2 models × 3 templates × 3 difficulties = 18)
Right panel — per-experiment B1 procedure (flowchart)

Output: results/b1_sweep/figures/sweep_diagram.png

Usage:
    python make_sweep_diagram.py
    python make_sweep_diagram.py --out slides/sweep_diagram.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle

# ── palette ──────────────────────────────────────────────────────────────────
C = {
    "apertus":  "#b42828",
    "llama33":  "#1f407a",
    "easy":     "#2e8b57",
    "medium":   "#d97b2a",
    "hard":     "#8b2252",
    "pair":     ("#cce5ff", "#2471a3"),
    "ab":       ("#d4edda", "#198754"),
    "ba":       ("#fff3cd", "#856404"),
    "judge":    ("#e2d9f3", "#5b2c8e"),
    "label":    ("#f8d7da", "#922b21"),
    "compare":  ("#e8f5e9", "#1e7e34"),
    "metrics":  ("#fef9e7", "#b7950b"),
}
MODELS      = [("apertus", "Apertus-70B"), ("llama33", "Llama-3.3-70B")]
TEMPLATES   = ["expert_rater", "llm_judge", "opro"]
TMPL_DISPLAY = {"expert_rater": "expert\nrater", "llm_judge": "llm\njudge", "opro": "opro"}
DIFFICULTIES = ["easy", "medium", "hard"]


# ── shared helpers ────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, text, fc, ec, fs=9, bold=False, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3, clip_on=False,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#111", zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#555", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=2)


# ── left panel: sweep matrix ──────────────────────────────────────────────────
def draw_matrix(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # grid geometry
    gx0, gx1 = 0.23, 0.98
    gy0, gy1 = 0.13, 0.82
    cw = (gx1 - gx0) / 3
    ch = (gy1 - gy0) / 3
    col_cx = [gx0 + cw * (i + 0.5) for i in range(3)]
    row_cy = [gy1 - ch * (i + 0.5) for i in range(3)]

    # ── title & subtitle
    ax.text(0.60, 0.96, "B1 Position Bias — Sweep Design",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111")
    ax.text(0.60, 0.91,
            "2 models  ×  3 templates  ×  3 difficulties  =  18 experiments",
            ha="center", va="center", fontsize=10.5, color="#555")

    # ── column headers (difficulties)
    for ci, diff in enumerate(DIFFICULTIES):
        ax.text(col_cx[ci], gy1 + 0.055, diff.capitalize(),
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=C[diff], edgecolor="none"))

    # ── row headers (templates) — left of grid
    for ri, tmpl in enumerate(TEMPLATES):
        ax.text(gx0 - 0.025, row_cy[ri], TMPL_DISPLAY[tmpl],
                ha="right", va="center", fontsize=9.5, fontweight="bold",
                color="#333",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#eef2f5",
                          edgecolor="#adb5bd", linewidth=1.2))

    # ── grid cells
    sq   = cw * 0.22    # model-square side
    gap  = sq * 1.35    # spacing between the two squares
    for ri in range(3):
        for ci in range(3):
            cx, cy = col_cx[ci], row_cy[ri]

            # cell background
            ax.add_patch(Rectangle(
                (cx - cw / 2, cy - ch / 2), cw, ch,
                facecolor="#f4f6f8", edgecolor="#ced4da", linewidth=1.0, zorder=1,
            ))

            # two model squares (Ap | Ll)
            for mi, (mk, _) in enumerate(MODELS):
                mx = cx + (mi - 0.5) * gap
                my = cy + 0.025
                ax.add_patch(FancyBboxPatch(
                    (mx - sq / 2, my - sq / 2), sq, sq,
                    boxstyle="round,pad=0.006",
                    facecolor=C[mk], edgecolor="white", linewidth=0.7, zorder=2,
                ))
                short = "Ap" if mk == "apertus" else "Ll"
                ax.text(mx, my, short, ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold", zorder=3)

            # sub-caption
            ax.text(cx, cy - ch * 0.32, "2 judge models", ha="center", va="center",
                    fontsize=7, color="#888", style="italic", zorder=3)

    # ── axis labels
    ax.text(0.60, 0.065, "difficulty →",
            ha="center", va="center", fontsize=9, color="#777")
    ax.text(0.075, 0.475, "template →",
            ha="center", va="center", fontsize=9, color="#777", rotation=90)

    # ── legend
    ax.text(0.37, 0.035, "Models:", ha="center", va="center",
            fontsize=9, color="#444")
    for mi, (mk, ml) in enumerate(MODELS):
        lx = 0.47 + mi * 0.29
        ax.add_patch(FancyBboxPatch(
            (lx - 0.018, 0.025 - 0.012), 0.036, 0.024,
            boxstyle="round,pad=0.004",
            facecolor=C[mk], edgecolor="white", linewidth=0.5, zorder=3,
        ))
        ax.text(lx + 0.03, 0.035, ml, ha="left", va="center",
                fontsize=9, color="#333")


# ── right panel: B1 flowchart ─────────────────────────────────────────────────
def draw_flowchart(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── title
    ax.text(0.50, 0.96, "B1: Per-experiment procedure",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#111")
    ax.text(0.50, 0.91, "(same procedure for all 18 combinations)",
            ha="center", va="center", fontsize=9, color="#777", style="italic")

    # ── response pair
    box(ax, 0.50, 0.83, 0.55, 0.085, "Response Pair\n(gold label known)",
        *C["pair"], fs=9.5, bold=True)

    # ── split arrows to AB / BA
    arrow(ax, 0.50, 0.788, 0.23, 0.723, color="#555")
    arrow(ax, 0.50, 0.788, 0.77, 0.723, color="#555")

    # ── order boxes
    box(ax, 0.23, 0.685, 0.38, 0.08, "[Response A]  [Response B]\n        AB order",
        *C["ab"], fs=8.5)
    box(ax, 0.77, 0.685, 0.38, 0.08, "[Response B]  [Response A]\n        BA order (flipped)",
        *C["ba"], fs=8.5)

    # note: what "correct" means per ordering
    ax.text(0.23, 0.627, "correct answer: A", ha="center", va="center",
            fontsize=7.5, color="#198754", style="italic")
    ax.text(0.77, 0.627, "correct answer: B", ha="center", va="center",
            fontsize=7.5, color="#856404", style="italic")

    # ── arrows to judge
    arrow(ax, 0.23, 0.615, 0.23, 0.563)
    arrow(ax, 0.77, 0.615, 0.77, 0.563)

    # ── judge boxes
    box(ax, 0.23, 0.525, 0.37, 0.077, "LLM Judge\n(model + template)",
        *C["judge"], fs=8.5)
    box(ax, 0.77, 0.525, 0.37, 0.077, "LLM Judge\n(model + template)",
        *C["judge"], fs=8.5)

    # link note between judge boxes
    ax.annotate("", xy=(0.59, 0.525), xytext=(0.41, 0.525),
                arrowprops=dict(arrowstyle="<->", color="#aaa", lw=1.2))
    ax.text(0.50, 0.543, "same combo", ha="center", va="bottom",
            fontsize=7, color="#aaa", style="italic")

    # ── arrows to labels
    arrow(ax, 0.23, 0.487, 0.23, 0.435)
    arrow(ax, 0.77, 0.487, 0.77, 0.435)

    # ── label output boxes
    box(ax, 0.23, 0.400, 0.29, 0.07, "Label:  A / B / C",
        *C["label"], fs=9, bold=True)
    box(ax, 0.77, 0.400, 0.29, 0.07, "Label:  A / B / C",
        *C["label"], fs=9, bold=True)

    # ── converging arrows to compare
    arrow(ax, 0.23, 0.365, 0.42, 0.295, color="#555")
    arrow(ax, 0.77, 0.365, 0.58, 0.295, color="#555")

    # ── compare box
    box(ax, 0.50, 0.258, 0.52, 0.075, "Compare labels\n(flip BA label before comparing)",
        *C["compare"], fs=8.5, bold=True)

    # ── arrow to metrics
    arrow(ax, 0.50, 0.221, 0.50, 0.172)

    # ── metrics box
    metrics_txt = (
        "position_consistency    position_bias_rate\n"
        "ab_accuracy    ba_accuracy    accuracy_gap"
    )
    box(ax, 0.50, 0.120, 0.88, 0.095, metrics_txt, *C["metrics"], fs=8.5)

    # ── repeat note at bottom
    ax.text(0.50, 0.030,
            "Repeated for each of the 18 (model, template, difficulty) combinations",
            ha="center", va="center", fontsize=8.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#ccc", linewidth=1.0))


# ── main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: results/b1_sweep/figures/sweep_diagram.png)")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out or Path("results/b1_sweep/figures/sweep_diagram.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(18, 8),
        gridspec_kw={"width_ratios": [1.05, 0.95], "wspace": 0.05},
    )
    fig.patch.set_facecolor("white")

    # thin divider between panels
    fig.add_artist(plt.Line2D(
        [0.505, 0.505], [0.04, 0.96],
        transform=fig.transFigure, color="#ddd", linewidth=1.2,
    ))

    draw_matrix(ax_l)
    draw_flowchart(ax_r)

    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
