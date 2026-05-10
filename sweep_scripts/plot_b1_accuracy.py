"""
plot_b1_accuracy.py
-------------------
Stacked-bar accuracy plots for the B1 position-bias sweep.

Accuracy is partitioned by where the correct answer physically sits in the pair:
  ab_accuracy — correct answer was placed in position A (first)
  ba_accuracy — correct answer was placed in position B (second)
  c_accuracy  — gold label is a tie (C)

Each bar is stacked to 1.0:  correct (bottom) | error-type-1 | error-type-2
This shows accuracy AND the composition of mistakes in a single glyph.

Saved to <out>/ (default results/b1_sweep/figures/accuracy/):
  per_difficulty/{easy,medium,hard}.png   — fixed difficulty, all model × template
  per_model/{apertus,llama33,glm47}.png   — fixed model, all difficulty × template
  per_template/{expert_rater,...}.png     — fixed template, all difficulty × model
  combined.png                            — 3×3 grid (acc type × difficulty)

Usage:
    python sweep_scripts/plot_b1_accuracy.py
    python sweep_scripts/plot_b1_accuracy.py --root results/b1_sweep --out results/figs
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config import (
    MODELS_ORDER, MODEL_LABELS, MODEL_HATCHES,
    TEMPLATES_ORDER, TEMPLATE_LABELS, TEMPLATE_HATCHES,
    DIFFICULTIES, ACC_TYPES,
)

# ── accuracy colour scheme ────────────────────────────────────────────────────
# Each acc type has 3 segment colours: [correct, error_key_1, error_key_2]
# AB: correct=green, wrong→B=orange, wrong→C=gray
# BA: correct=blue,  wrong→A=orange, wrong→C=gray
# C:  correct=purple, wrong→A=green, wrong→B=red
ACC_SEGMENT_COLORS = {
    "ab": ["#27ae60", "#e67e22", "#95a5a6"],
    "ba": ["#2980b9", "#e67e22", "#95a5a6"],
    "c":  ["#8e44ad", "#27ae60", "#e74c3c"],
}
ACC_ERR_KEYS = {
    "ab": ["B", "C"],
    "ba": ["A", "C"],
    "c":  ["A", "B"],
}
ACC_YLABELS = {
    "ab": "Primacy accuracy\n(correct in slot A)",
    "ba": "Recency accuracy\n(correct in slot B)",
    "c":  "Tie accuracy\n(gold = tie)",
}

# ── bar layout ────────────────────────────────────────────────────────────────
BAR_W     = 0.22
INTRA_GAP = 0.04   # gap between bars within a group
INTER_GAP = 0.38   # extra space between groups


# ── data loading ──────────────────────────────────────────────────────────────
def load_metrics(root: Path) -> dict:
    """Return nested dict: metrics[model][template][difficulty] = record."""
    out: dict = {}
    for path in sorted(root.glob("*/*/*/metrics.json")):
        p = path.parts
        model, template, difficulty = p[-4], p[-3], p[-2]
        with open(path) as f:
            out.setdefault(model, {}).setdefault(template, {})[difficulty] = json.load(f)
    return out


def _build_resolved_map(metrics: dict) -> dict:
    """Return {(model, template): resolved_template} from _meta in each record."""
    out = {}
    for model, tmpl_dict in metrics.items():
        for tmpl, diff_dict in tmpl_dict.items():
            for rec in diff_dict.values():
                meta = rec.get("_meta", {})
                out[(model, tmpl)] = meta.get("resolved_template", tmpl)
    return out


_MISSING = object()


def _get(rec, *keys):
    """Safe nested dict access; returns None if any key is absent."""
    d = rec
    for k in keys:
        if not isinstance(d, dict):
            return None
        nxt = d.get(k, _MISSING)
        if nxt is _MISSING:
            return None
        d = nxt
    return d


# ── layout helpers ────────────────────────────────────────────────────────────
def _group_positions(n_groups: int, n_per_group: int):
    """Return (flat x positions, group centre x positions) for grouped bars."""
    step       = BAR_W + INTRA_GAP
    span       = n_per_group * step - INTRA_GAP   # width of one group of bars
    group_step = span + INTER_GAP                 # distance between group left edges
    xs, centers = [], []
    for g in range(n_groups):
        left = g * group_step
        centers.append(left + (n_per_group - 1) * step / 2)
        for b in range(n_per_group):
            xs.append(left + b * step)
    return xs, centers


# ── core drawing ──────────────────────────────────────────────────────────────
def _stacked_bar(ax, x: float, rec: dict | None, acc_type: str, hatch: str = "") -> None:
    """Draw one stacked accuracy bar at position x for the given acc_type."""
    if rec is None:
        return
    acc      = _get(rec, "accuracy", f"{acc_type}_accuracy")
    err_dist = _get(rec, "accuracy", f"{acc_type}_error_distribution")

    if acc is None or (isinstance(acc, float) and np.isnan(acc)):
        return

    colors   = ACC_SEGMENT_COLORS[acc_type]
    err_keys = ACC_ERR_KEYS[acc_type]

    # Bottom segment: correct predictions
    ax.bar(x, acc, BAR_W, bottom=0.0, color=colors[0],
           hatch=hatch, edgecolor="black", linewidth=0.5, zorder=3)

    bottom = float(acc)
    if isinstance(err_dist, dict) and err_dist:
        # New-format metrics: split error by label
        for j, ek in enumerate(err_keys):
            val = (1.0 - acc) * err_dist.get(ek, 0.0)
            if val > 5e-4:
                ax.bar(x, val, BAR_W, bottom=bottom,
                       color=colors[j + 1], edgecolor="black", linewidth=0.4, zorder=3)
            bottom += val
    elif acc < 1.0:
        # Old-format metrics: show total error mass without breakdown
        ax.bar(x, 1.0 - acc, BAR_W, bottom=bottom,
               color="#cccccc", edgecolor="black", linewidth=0.4, zorder=3)


def _style_row(ax, acc_type: str, tick_pos: list, tick_labels: list) -> None:
    """Apply shared styling and an inset colour legend to one accuracy-type row."""
    ax.set_ylim(0, 1.15)
    ax.axhline(1 / 3, color="gray", linewidth=0.8, linestyle=":", zorder=0,
               label="Random (1/3)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel(ACC_YLABELS[acc_type], fontsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Inset legend explaining segment colours for this acc type
    clrs = ACC_SEGMENT_COLORS[acc_type]
    ek   = ACC_ERR_KEYS[acc_type]
    handles = [
        mpatches.Patch(fc=clrs[0], ec="black", label="Correct"),
        mpatches.Patch(fc=clrs[1], ec="black", label=f"Wrong → {ek[0]}"),
        mpatches.Patch(fc=clrs[2], ec="black", label=f"Wrong → {ek[1]}"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.85,
              handlelength=1.2, borderpad=0.5)


def _template_legend_handles(resolved_map: dict | None = None,
                              model: str | None = None) -> list:
    """Legend patches for templates.  When model + resolved_map are given,
    appends '→<resolved>' to the label for templates that differ per model."""
    handles = []
    for t in TEMPLATES_ORDER:
        label = TEMPLATE_LABELS.get(t, t)
        if resolved_map is not None and model is not None:
            resolved = resolved_map.get((model, t), t)
            if resolved != t:
                label += f" →{resolved}"
        handles.append(mpatches.Patch(fc="white", hatch=TEMPLATE_HATCHES[t],
                                      ec="black", label=label))
    return handles


def _model_legend_handles() -> list:
    return [
        mpatches.Patch(fc="white", hatch=MODEL_HATCHES[m], ec="black",
                       label=MODEL_LABELS[m])
        for m in MODELS_ORDER
    ]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── per-difficulty ────────────────────────────────────────────────────────────
def plot_per_difficulty(metrics: dict, out: Path) -> None:
    """
    One figure per difficulty level.
    Rows = accuracy type (AB / BA / C).
    x groups = model,  bars within group = template (hatch-encoded).
    """
    xs, centers = _group_positions(len(MODELS_ORDER), len(TEMPLATES_ORDER))
    group_labels = [MODEL_LABELS[m] for m in MODELS_ORDER]

    for diff in DIFFICULTIES:
        fig, axes = plt.subplots(len(ACC_TYPES), 1, figsize=(9, 3 * len(ACC_TYPES)), sharex=True)
        fig.suptitle(f"B1 Accuracy — {diff.capitalize()} difficulty",
                     fontsize=13, fontweight="bold")

        for row, acc_type in enumerate(ACC_TYPES):
            ax = axes[row]
            for gi, model in enumerate(MODELS_ORDER):
                for bi, template in enumerate(TEMPLATES_ORDER):
                    xi = gi * len(TEMPLATES_ORDER) + bi
                    rec = metrics.get(model, {}).get(template, {}).get(diff)
                    _stacked_bar(ax, xs[xi], rec, acc_type,
                                 hatch=TEMPLATE_HATCHES[template])
            _style_row(ax, acc_type, centers, group_labels)

        fig.legend(handles=_template_legend_handles(), loc="lower center", ncol=3,
                   fontsize=8, bbox_to_anchor=(0.5, -0.01), frameon=True,
                   title="Template (hatch)")
        fig.tight_layout()
        _save(fig, out / "per_difficulty" / f"{diff}.png")


# ── per-model ─────────────────────────────────────────────────────────────────
def plot_per_model(metrics: dict, out: Path,
                   resolved_map: dict | None = None) -> None:
    """
    One figure per model.
    Rows = accuracy type.
    x groups = difficulty,  bars within group = template (hatch-encoded).
    """
    xs, centers = _group_positions(len(DIFFICULTIES), len(TEMPLATES_ORDER))
    group_labels = [d.capitalize() for d in DIFFICULTIES]

    for model in MODELS_ORDER:
        fig, axes = plt.subplots(len(ACC_TYPES), 1, figsize=(9, 3 * len(ACC_TYPES)), sharex=True)
        fig.suptitle(f"B1 Accuracy — {MODEL_LABELS[model]}",
                     fontsize=13, fontweight="bold")

        for row, acc_type in enumerate(ACC_TYPES):
            ax = axes[row]
            for gi, diff in enumerate(DIFFICULTIES):
                for bi, template in enumerate(TEMPLATES_ORDER):
                    xi = gi * len(TEMPLATES_ORDER) + bi
                    rec = metrics.get(model, {}).get(template, {}).get(diff)
                    _stacked_bar(ax, xs[xi], rec, acc_type,
                                 hatch=TEMPLATE_HATCHES[template])
            _style_row(ax, acc_type, centers, group_labels)

        fig.legend(handles=_template_legend_handles(resolved_map, model),
                   loc="lower center", ncol=len(TEMPLATES_ORDER),
                   fontsize=8, bbox_to_anchor=(0.5, -0.01), frameon=True,
                   title="Template (hatch)")
        fig.tight_layout()
        _save(fig, out / "per_model" / f"{model}.png")


# ── per-template ──────────────────────────────────────────────────────────────
def plot_per_template(metrics: dict, out: Path,
                      resolved_map: dict | None = None) -> None:
    """
    One figure per template.
    Rows = accuracy type.
    x groups = difficulty,  bars within group = model (hatch-encoded).
    """
    xs, centers = _group_positions(len(DIFFICULTIES), len(MODELS_ORDER))
    group_labels = [d.capitalize() for d in DIFFICULTIES]

    for template in TEMPLATES_ORDER:
        # Build a subtitle note when models use different resolved prompts
        if resolved_map is not None:
            resolutions = {m: resolved_map.get((m, template), template)
                           for m in MODELS_ORDER}
            unique = set(resolutions.values())
            if len(unique) > 1:
                parts = [f"{MODEL_LABELS[m]}→{resolutions[m]}"
                         for m in MODELS_ORDER if resolutions[m] != template]
                resolve_note = f"\n({', '.join(parts)})"
            elif len(unique) == 1 and list(unique)[0] != template:
                resolve_note = f"\n(→{list(unique)[0]})"
            else:
                resolve_note = ""
        else:
            resolve_note = ""

        fig, axes = plt.subplots(len(ACC_TYPES), 1, figsize=(9, 3 * len(ACC_TYPES)), sharex=True)
        fig.suptitle(f"B1 Accuracy — {TEMPLATE_LABELS.get(template, template)}{resolve_note}",
                     fontsize=13, fontweight="bold")

        for row, acc_type in enumerate(ACC_TYPES):
            ax = axes[row]
            for gi, diff in enumerate(DIFFICULTIES):
                for bi, model in enumerate(MODELS_ORDER):
                    xi = gi * len(MODELS_ORDER) + bi
                    rec = metrics.get(model, {}).get(template, {}).get(diff)
                    _stacked_bar(ax, xs[xi], rec, acc_type,
                                 hatch=MODEL_HATCHES[model])
            _style_row(ax, acc_type, centers, group_labels)

        fig.legend(handles=_model_legend_handles(), loc="lower center", ncol=3,
                   fontsize=8, bbox_to_anchor=(0.5, -0.01), frameon=True,
                   title="Model (hatch)")
        fig.tight_layout()
        _save(fig, out / "per_template" / f"{template}.png")


# ── combined overview ─────────────────────────────────────────────────────────
def plot_combined(metrics: dict, out: Path) -> None:
    """
    3×3 grid: rows = accuracy type (AB / BA / C), cols = difficulty.
    Each cell: 9 stacked bars (3 model groups × 3 template bars).
    Lets you compare everything at a glance in a single figure.
    """
    xs, centers = _group_positions(len(MODELS_ORDER), len(TEMPLATES_ORDER))
    group_labels = [MODEL_LABELS[m] for m in MODELS_ORDER]

    fig, axes = plt.subplots(len(ACC_TYPES), len(DIFFICULTIES),
                             figsize=(6 * len(DIFFICULTIES), 4 * len(ACC_TYPES)),
                             sharex="col", sharey="row")
    fig.suptitle("B1 Accuracy — Combined Overview\n"
                 "(rows: accuracy type  |  cols: difficulty  |  "
                 "groups: model  |  bars: template)",
                 fontsize=12, fontweight="bold")

    for row, acc_type in enumerate(ACC_TYPES):
        for col, diff in enumerate(DIFFICULTIES):
            ax = axes[row][col]

            for gi, model in enumerate(MODELS_ORDER):
                for bi, template in enumerate(TEMPLATES_ORDER):
                    xi = gi * len(TEMPLATES_ORDER) + bi
                    rec = metrics.get(model, {}).get(template, {}).get(diff)
                    _stacked_bar(ax, xs[xi], rec, acc_type,
                                 hatch=TEMPLATE_HATCHES[template])

            ax.set_ylim(0, 1.15)
            ax.axhline(1 / 3, color="gray", linewidth=0.7, linestyle=":", zorder=0)
            ax.grid(axis="y", alpha=0.2, zorder=0)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xticks(centers)
            ax.set_xticklabels(group_labels, fontsize=7.5)

            if col == 0:
                ax.set_ylabel(ACC_YLABELS[acc_type], fontsize=8)
            if row == 0:
                ax.set_title(diff.capitalize(), fontsize=10, fontweight="bold")

            # Compact per-cell segment legend
            clrs = ACC_SEGMENT_COLORS[acc_type]
            ek   = ACC_ERR_KEYS[acc_type]
            cell_handles = [
                mpatches.Patch(fc=clrs[0], ec="black", label="Correct"),
                mpatches.Patch(fc=clrs[1], ec="black", label=f"→{ek[0]}"),
                mpatches.Patch(fc=clrs[2], ec="black", label=f"→{ek[1]}"),
            ]
            ax.legend(handles=cell_handles, fontsize=6, loc="upper right",
                      framealpha=0.8, handlelength=1.0, borderpad=0.3)

    fig.legend(handles=_template_legend_handles(), loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01), frameon=True,
               title="Template (hatch)")
    fig.tight_layout()
    _save(fig, out / "combined.png")


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("results/b1_sweep"),
                   help="Root directory written by run_b1_sweep.py")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: <root>/figures/accuracy)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.out or (args.root / "figures" / "accuracy")

    metrics = load_metrics(args.root)
    if not metrics:
        print(f"No metrics.json files found under {args.root}. "
              "Run run_b1_sweep.py first.")
        return

    total = sum(
        1
        for m in metrics.values()
        for t in m.values()
        for _ in t.values()
    )
    print(f"Loaded {total} experiment results.")

    resolved_map = _build_resolved_map(metrics)

    plot_per_difficulty(metrics, out)
    plot_per_model(metrics, out, resolved_map=resolved_map)
    plot_per_template(metrics, out, resolved_map=resolved_map)
    plot_combined(metrics, out)
    print(f"\nAll accuracy figures saved to {out}/")


if __name__ == "__main__":
    main()
