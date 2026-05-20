"""
run_b4_analyze.py
-----------------
Post-hoc analysis for the B4 paraphrase-sensitivity experiment.

Reads   results/b4_paraphrase_sensitivity/<model_tag>.jsonl
        data/helpsteer2_{split}_paraphrased.json   (for paraphrase quality)
Writes  results/b4_paraphrase_sensitivity/plots/*.png
        results/b4_paraphrase_sensitivity/full_metrics.json

Judge-sensitivity figures
-------------------------
  fig1_overview.png          — 6-panel headline metrics (both models)
  fig1b_bucket_overview.png  — same 6 panels but x-axis = difficulty bucket
  fig2_bucket_flip.png       — flip rate per difficulty bucket
  fig3_bucket_accuracy.png   — accuracy original / para / delta per bucket
  fig4_bucket_position.png   — position bias rate per bucket
  fig5_bucket_primacy.png    — primacy + recency rate per bucket
  fig6_label_distribution.png— label distribution (A/B/C) per condition
  fig7_bias_decomposition.png— stacked AB-vs-BA decomposition
  fig8_volatile_pairs.png    — volatile pair counts per difficulty bucket
  fig9_score_gap_flip.png    — flip rate binned by score_gap
  fig10_summary_table.png    — full numeric summary table

Paraphrase-quality figures  (needs --data-path, sentence-transformers, rouge-score)
----------------------------
  fig_pq_distributions.png   — histogram of cosine sim and ROUGE-L
  fig_pq_scatter.png         — cosine sim vs ROUGE-L scatter coloured by difficulty
  fig_pq_by_bucket.png       — mean cosine sim / ROUGE-L / flagged rate per bucket
  fig_pq_length.png          — length ratio (para/orig) distribution + per bucket
  fig_pq_table.png           — summary table with mean±std, min/max, flagged %

Usage
-----
    python run_b4_analyze.py
    python run_b4_analyze.py --results-dir results/b4_paraphrase_sensitivity
    python run_b4_analyze.py --data-path data/helpsteer2_validation_paraphrased.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional heavy deps — gracefully skipped if not installed
try:
    from sentence_transformers import SentenceTransformer as _ST
    _HAVE_ST = True
except ImportError:
    _HAVE_ST = False

try:
    from rouge_score import rouge_scorer as _rs
    _HAVE_ROUGE = True
except ImportError:
    _HAVE_ROUGE = False

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
})

DIFFICULTY_ORDER = ("tie", "hard", "medium", "easy")
DIFF_COLORS = {
    "tie":    "#d62728",
    "hard":   "#ff7f0e",
    "medium": "#2ca02c",
    "easy":   "#1f77b4",
}

MODEL_TAGS = {
    "meta_llama_llama_3_3_70b_instruct": "Llama-3.3-70B",
    "swiss_ai_apertus_70b_instruct_2509": "Apertus-70B",
}
MODEL_COLORS = {
    "Llama-3.3-70B": "#4C72B0",
    "Apertus-70B":   "#DD8452",
}
MODEL_ORDER = ["Llama-3.3-70B", "Apertus-70B"]
FLIP_MAP = {"A": "B", "B": "A", "C": "C"}
CONDITIONS = ("original_AB", "original_BA", "para_AB", "para_BA")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(ln) for ln in path.open()])


def load_all(results_dir: Path) -> dict[str, pd.DataFrame]:
    dfs = {}
    for tag, name in MODEL_TAGS.items():
        p = results_dir / f"{tag}.jsonl"
        if p.exists():
            dfs[name] = load_jsonl(p)
        else:
            print(f"  WARNING: {p} not found — skipping {name}")
    return dfs


# ── Metric primitives ─────────────────────────────────────────────────────────

def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per prompt_id, one column per condition label."""
    wide = df.pivot_table(index="prompt_id", columns="condition",
                          values="label", aggfunc="first")
    extra = [c for c in ("gold_label", "difficulty", "score_gap") if c in df.columns]
    meta  = df.drop_duplicates("prompt_id").set_index("prompt_id")[extra]
    return wide.join(meta)


def _pos_stats(w: pd.DataFrame, ab: str, ba: str) -> dict:
    """Position-consistency decomposition for a pair of conditions."""
    _nan = {"position_consistency": np.nan, "position_bias_rate": np.nan,
            "primacy_bias_rate": np.nan, "recency_bias_rate": np.nan,
            "other_rate": np.nan, "n": 0}
    if ab not in w.columns or ba not in w.columns:
        return _nan
    sub = w[[ab, ba]].dropna()
    if sub.empty:
        return _nan
    n            = len(sub)
    consistent   = sub.apply(lambda r: FLIP_MAP.get(r[ab]) == r[ba], axis=1).sum()
    primacy_bias = ((sub[ab] == "A") & (sub[ba] == "A")).sum()
    recency_bias = ((sub[ab] == "B") & (sub[ba] == "B")).sum()
    other        = n - consistent - primacy_bias - recency_bias
    return {
        "position_consistency": consistent / n,
        "position_bias_rate":   1 - consistent / n,
        "primacy_bias_rate":    primacy_bias / n,
        "recency_bias_rate":    recency_bias / n,
        "other_rate":           other / n,
        "n": n,
    }


def _label_rates(series: pd.Series) -> dict:
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {"A": np.nan, "B": np.nan, "C": np.nan, "n": 0}
    return {"A": (s == "A").sum() / n,
            "B": (s == "B").sum() / n,
            "C": (s == "C").sum() / n,
            "n": n}


def _acc(w: pd.DataFrame, col: str) -> float:
    if col not in w.columns or "gold_label" not in w.columns:
        return np.nan
    sub = w[[col, "gold_label"]].dropna()
    return float((sub[col] == sub["gold_label"]).mean()) if len(sub) else np.nan


def _flip_rate(w: pd.DataFrame) -> tuple[float, int]:
    if "original_AB" not in w.columns or "para_AB" not in w.columns:
        return np.nan, 0
    sub = w[["original_AB", "para_AB"]].dropna()
    return (float((sub["original_AB"] != sub["para_AB"]).mean()) if len(sub) else np.nan), len(sub)


def _cohens_kappa(w: pd.DataFrame, col_a: str, col_b: str) -> float:
    """Cohen's kappa between two label columns (multi-class A/B/C)."""
    if col_a not in w.columns or col_b not in w.columns:
        return np.nan
    sub = w[[col_a, col_b]].dropna()
    if len(sub) == 0:
        return np.nan
    p_o = float((sub[col_a] == sub[col_b]).mean())
    p_e = sum(
        (sub[col_a] == k).mean() * (sub[col_b] == k).mean()
        for k in ("A", "B", "C")
    )
    return float((p_o - p_e) / (1 - p_e)) if (1 - p_e) > 1e-9 else np.nan


def _snorm(f_para: float, f_pos: float) -> float:
    """Normalised sensitivity S_norm = F_para / F_pos."""
    if math.isnan(f_para) or math.isnan(f_pos) or f_pos < 1e-9:
        return np.nan
    return float(f_para / f_pos)


# ── Full metric computation ───────────────────────────────────────────────────

def _metrics_for_slice(w: pd.DataFrame) -> dict:
    """All metrics for a (possibly filtered) wide DataFrame."""
    fr, n_valid = _flip_rate(w)
    acc_orig = _acc(w, "original_AB")
    acc_para = _acc(w, "para_AB")
    acc_delta = (acc_para - acc_orig
                 if not (math.isnan(acc_orig) or math.isnan(acc_para)) else np.nan)

    pos_orig = _pos_stats(w, "original_AB", "original_BA")
    pos_para = _pos_stats(w, "para_AB",     "para_BA")
    pb_delta = (pos_para["position_bias_rate"] - pos_orig["position_bias_rate"]
                if not (math.isnan(pos_orig["position_bias_rate"])
                        or math.isnan(pos_para["position_bias_rate"])) else np.nan)

    lr = {c: _label_rates(w[c]) if c in w.columns else {"A": np.nan, "B": np.nan, "C": np.nan, "n": 0}
          for c in CONDITIONS}

    # Volatile pairs by difficulty
    vol_diff: dict[str, int] = {}
    if "difficulty" in w.columns and n_valid > 0:
        sv = w[["original_AB", "para_AB", "difficulty"]].dropna(
            subset=["original_AB", "para_AB"])
        vol = sv[sv["original_AB"] != sv["para_AB"]]
        vol_diff = (vol.groupby("difficulty").size()
                    .reindex(list(DIFFICULTY_ORDER), fill_value=0).to_dict())

    kappa   = _cohens_kappa(w, "original_AB", "para_AB")
    f_pos   = pos_orig.get("position_bias_rate", np.nan)
    s_norm  = _snorm(fr, f_pos)

    return {
        "n": len(w),
        "input_sensitivity": {
            "flip_rate":  fr,
            "agreement":  1 - fr if not math.isnan(fr) else np.nan,
            "kappa":      kappa,
            "f_pos":      f_pos,
            "s_norm":     s_norm,
            "n_valid":    n_valid,
        },
        "accuracy": {
            "original": acc_orig,
            "para":     acc_para,
            "delta":    acc_delta,
        },
        "label_rates": lr,
        "position_consistency": {
            "original":            pos_orig,
            "para":                pos_para,
            "position_bias_delta": pb_delta,
        },
        "volatile_by_difficulty": vol_diff,
    }


def compute_all_metrics(df: pd.DataFrame) -> dict:
    """Overall + per-bucket metrics for one model."""
    w       = _wide(df)
    overall = _metrics_for_slice(w)

    buckets: dict[str, dict] = {}
    if "difficulty" in w.columns:
        for d in DIFFICULTY_ORDER:
            sub = w[w["difficulty"] == d]
            if not sub.empty:
                buckets[d] = _metrics_for_slice(sub)

    overall["buckets"] = buckets
    return overall


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def _ann(ax: plt.Axes, bars, fmt: str = ".3f",
         pad_frac: float = 0.02, ylim: tuple | None = None) -> None:
    """Annotate bars with their value."""
    span = (ylim[1] - ylim[0]) if ylim else 1.0
    for b in bars:
        h = b.get_height()
        if math.isnan(h):
            continue
        ax.text(b.get_x() + b.get_width() / 2,
                h + pad_frac * span,
                f"{h:{fmt}}", ha="center", va="bottom", fontsize=8)


def _grouped_bars(ax: plt.Axes, x, groups: list[tuple], width: float = 0.35,
                  ylim: tuple | None = None, fmt: str = ".3f",
                  hline: float | None = None) -> None:
    """Generic grouped-bar helper. groups = [(label, values, color, hatch), ...]"""
    n = len(groups)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width
    for offset, (label, vals, color, hatch) in zip(offsets, groups):
        bars = ax.bar(x + offset, vals, width, color=color, hatch=hatch,
                      edgecolor="white", alpha=0.9, label=label)
        if ylim:
            _ann(ax, bars, fmt=fmt, ylim=ylim)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    if ylim:
        ax.set_ylim(*ylim)


# ── Figure 1: Overview ────────────────────────────────────────────────────────

def fig1_overview(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    colors = [MODEL_COLORS[m] for m in models]
    x = np.arange(len(models))
    w = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    def simple_bar(ax, vals, title, ylim=(0, 1), fmt=".3f", hline=None):
        bars = ax.bar(x, vals, 0.5, color=colors, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        _ann(ax, bars, fmt=fmt, ylim=ylim)
        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_ylim(*ylim)

    # [0,0] Flip rate
    vals = [metrics[m]["input_sensitivity"]["flip_rate"] for m in models]
    hi   = max(v for v in vals if not math.isnan(v)) * 1.4 + 0.05
    simple_bar(axes[0, 0], vals, "Flip Rate (original → para)", ylim=(0, max(hi, 0.5)))

    # [0,1] Accuracy original vs para (grouped)
    ax = axes[0, 1]
    for i, (key, label, hatch) in enumerate([("original", "Original", ""), ("para", "Para", "///")]):
        vals = [metrics[m]["accuracy"][key] for m in models]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=colors, hatch=hatch,
                      edgecolor="white", alpha=0.9, label=label)
        _ann(ax, bars, ylim=(0.3, 0.95))
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.3, 0.95); ax.set_title("Accuracy: Original vs Para", fontsize=10, pad=6)
    ax.legend(fontsize=8)

    # [0,2] Accuracy delta
    deltas = [metrics[m]["accuracy"]["delta"] for m in models]
    lo = min(min(deltas) - 0.04, -0.12)
    hi = max(max(deltas) + 0.04,  0.12)
    simple_bar(axes[0, 2], deltas, "Accuracy Delta (para − original)",
               ylim=(lo, hi), fmt="+.3f", hline=0)

    # [1,0] Position bias rate original vs para (grouped)
    ax = axes[1, 0]
    for i, (key, label, hatch) in enumerate([("original", "Original", ""), ("para", "Para", "///")]):
        vals = [metrics[m]["position_consistency"][key].get("position_bias_rate", np.nan)
                for m in models]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=colors, hatch=hatch,
                      edgecolor="white", alpha=0.9, label=label)
        _ann(ax, bars, ylim=(0, 0.7))
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 0.7); ax.set_title("Position Bias Rate", fontsize=10, pad=6)
    ax.legend(fontsize=8)

    # [1,1] Primacy rate original vs para (grouped)
    ax = axes[1, 1]
    for i, (cond, label, hatch) in enumerate([("original_AB", "Original", ""), ("para_AB", "Para", "///")]):
        vals = [metrics[m]["label_rates"][cond].get("A", np.nan) for m in models]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=colors, hatch=hatch,
                      edgecolor="white", alpha=0.9, label=label)
        _ann(ax, bars, ylim=(0, 0.8))
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 0.8); ax.set_title("Primacy Rate (A in AB condition)", fontsize=10, pad=6)
    ax.axhline(1 / 3, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label="random (1/3)")
    ax.legend(fontsize=8)

    # [1,2] Recency rate original vs para (grouped)
    ax = axes[1, 2]
    for i, (cond, label, hatch) in enumerate([("original_AB", "Original", ""), ("para_AB", "Para", "///")]):
        vals = [metrics[m]["label_rates"][cond].get("B", np.nan) for m in models]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=colors, hatch=hatch,
                      edgecolor="white", alpha=0.9, label=label)
        _ann(ax, bars, ylim=(0, 0.8))
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 0.8); ax.set_title("Recency Rate (B in AB condition)", fontsize=10, pad=6)
    ax.axhline(1 / 3, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)

    fig.suptitle("B4 Paraphrase Sensitivity — Overview", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig1_overview.png")


# ── Figure 1b: Overview per difficulty bucket ────────────────────────────────

def fig1b_bucket_overview(metrics: dict, plot_dir: Path) -> None:
    """Same 6 panels as fig1 but x-axis = difficulty bucket, lines/groups = model."""
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]
    x = np.arange(len(diffs))
    w = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def bucket_bar(ax, key_fn, title, ylim=(0, 1), fmt=".3f", hline=None):
        for i, model in enumerate(models):
            vals = [key_fn(metrics[model]["buckets"].get(d, {})) for d in diffs]
            bars = ax.bar(x + (i - 0.5) * w, vals, w,
                          color=MODEL_COLORS[model], edgecolor="white",
                          label=model, alpha=0.9)
            _ann(ax, bars, fmt=fmt, ylim=ylim)
        ax.set_xticks(x); ax.set_xticklabels(diffs)
        ax.set_ylim(*ylim); ax.set_title(title, fontsize=10, pad=6)
        ax.legend(fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    bucket_bar(axes[0, 0],
        lambda b: b.get("input_sensitivity", {}).get("flip_rate", np.nan),
        "Flip Rate per Bucket", ylim=(0, 0.7))

    # Accuracy orig vs para: two bar groups per bucket
    ax = axes[0, 1]
    for i, model in enumerate(models):
        for j, (acc_key, label, hatch) in enumerate([("original", "orig", ""), ("para", "para", "///")]):
            vals = [metrics[model]["buckets"].get(d, {}).get(
                        "accuracy", {}).get(acc_key, np.nan) for d in diffs]
            offset = (i * 2 + j - 1.5) * (w / 2)
            bars = ax.bar(x + offset, vals, w / 2,
                          color=MODEL_COLORS[model], hatch=hatch,
                          edgecolor="white", alpha=0.9,
                          label=f"{model} ({label})" if i == 0 or j == 0 else "")
            _ann(ax, bars, ylim=(0.2, 1.0))
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0.2, 1.0); ax.set_title("Accuracy (orig vs para) per Bucket", fontsize=10, pad=6)
    ax.legend(fontsize=7, ncol=2)

    bucket_bar(axes[0, 2],
        lambda b: b.get("accuracy", {}).get("delta", np.nan),
        "Accuracy Delta per Bucket", ylim=(-0.2, 0.2), fmt="+.3f", hline=0)

    bucket_bar(axes[1, 0],
        lambda b: b.get("position_consistency", {}).get("original", {}).get("position_bias_rate", np.nan),
        "Position Bias Rate (Original) per Bucket", ylim=(0, 0.8))

    bucket_bar(axes[1, 1],
        lambda b: b.get("label_rates", {}).get("original_AB", {}).get("A", np.nan),
        "Primacy Rate (Original AB) per Bucket", ylim=(0, 0.85))
    axes[1, 1].axhline(1 / 3, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

    bucket_bar(axes[1, 2],
        lambda b: b.get("label_rates", {}).get("original_AB", {}).get("B", np.nan),
        "Recency Rate (Original AB) per Bucket", ylim=(0, 0.85))
    axes[1, 2].axhline(1 / 3, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

    fig.suptitle("B4 Paraphrase Sensitivity — Overview per Difficulty Bucket",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig1b_bucket_overview.png")


# ── Figure 2: Flip rate per bucket ────────────────────────────────────────────

def fig2_bucket_flip(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(diffs))
    w = 0.35

    for i, model in enumerate(models):
        vals = [metrics[model]["buckets"].get(d, {}).get(
                    "input_sensitivity", {}).get("flip_rate", np.nan)
                for d in diffs]
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=model, alpha=0.9)
        _ann(ax, bars, ylim=(0, 0.65))

    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("Difficulty Bucket"); ax.set_ylabel("Flip Rate")
    ax.set_title("Flip Rate per Difficulty Bucket", fontsize=12, pad=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, plot_dir / "fig2_bucket_flip.png")


# ── Figure 3: Accuracy per bucket ─────────────────────────────────────────────

def fig3_bucket_accuracy(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(diffs))
    w = 0.35

    specs = [
        ("accuracy", "original", "Accuracy (Original)",        (0.3, 1.0),  ".3f",  None),
        ("accuracy", "para",     "Accuracy (Para)",             (0.3, 1.0),  ".3f",  None),
        ("accuracy", "delta",    "Accuracy Delta (para−orig)",  (-0.2, 0.2), "+.3f", 0),
    ]
    for ax, (outer, inner, title, ylim, fmt, hline) in zip(axes, specs):
        for i, model in enumerate(models):
            vals = [metrics[model]["buckets"].get(d, {}).get(outer, {}).get(inner, np.nan)
                    for d in diffs]
            bars = ax.bar(x + (i - 0.5) * w, vals, w,
                          color=MODEL_COLORS[model], edgecolor="white",
                          label=model, alpha=0.9)
            _ann(ax, bars, fmt=fmt, ylim=ylim)
        ax.set_xticks(x); ax.set_xticklabels(diffs)
        ax.set_ylim(*ylim); ax.set_title(title, fontsize=11, pad=6)
        ax.legend(fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    fig.suptitle("Accuracy by Difficulty Bucket", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig3_bucket_accuracy.png")


# ── Figure 4: Position bias per bucket ───────────────────────────────────────

def fig4_bucket_position(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(diffs))
    w = 0.35

    specs = [
        ("original", "position_bias_rate",   "Position Bias Rate (Original)",    (0, 0.75), ".3f",  None),
        ("para",     "position_bias_rate",   "Position Bias Rate (Para)",         (0, 0.75), ".3f",  None),
        (None,       "position_bias_delta",  "Position Bias Delta (para−orig)",   (-0.2, 0.2), "+.3f", 0),
    ]
    for ax, (cond, key, title, ylim, fmt, hline) in zip(axes, specs):
        for i, model in enumerate(models):
            if cond is None:
                vals = [metrics[model]["buckets"].get(d, {}).get(
                            "position_consistency", {}).get(key, np.nan)
                        for d in diffs]
            else:
                vals = [metrics[model]["buckets"].get(d, {}).get(
                            "position_consistency", {}).get(cond, {}).get(key, np.nan)
                        for d in diffs]
            bars = ax.bar(x + (i - 0.5) * w, vals, w,
                          color=MODEL_COLORS[model], edgecolor="white",
                          label=model, alpha=0.9)
            _ann(ax, bars, fmt=fmt, ylim=ylim)
        ax.set_xticks(x); ax.set_xticklabels(diffs)
        ax.set_ylim(*ylim); ax.set_title(title, fontsize=11, pad=6)
        ax.legend(fontsize=8)
        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    fig.suptitle("Position Bias by Difficulty Bucket", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig4_bucket_position.png")


# ── Figure 5: Primacy & recency per bucket ────────────────────────────────────

def fig5_bucket_primacy(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    specs = [
        ("original_AB", "A", "Primacy Rate — Original AB",  (0, 0.85)),
        ("para_AB",     "A", "Primacy Rate — Para AB",       (0, 0.85)),
        ("original_AB", "B", "Recency Rate — Original AB",  (0, 0.85)),
        ("para_AB",     "B", "Recency Rate — Para AB",       (0, 0.85)),
    ]
    x = np.arange(len(diffs))
    w = 0.35

    for ax, (cond, lbl, title, ylim) in zip(axes.flatten(), specs):
        for i, model in enumerate(models):
            vals = [metrics[model]["buckets"].get(d, {}).get(
                        "label_rates", {}).get(cond, {}).get(lbl, np.nan)
                    for d in diffs]
            bars = ax.bar(x + (i - 0.5) * w, vals, w,
                          color=MODEL_COLORS[model], edgecolor="white",
                          label=model, alpha=0.9)
            _ann(ax, bars, ylim=ylim)
        ax.set_xticks(x); ax.set_xticklabels(diffs)
        ax.set_ylim(*ylim); ax.set_title(title, fontsize=11, pad=6)
        ax.axhline(1 / 3, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label="random")
        ax.legend(fontsize=8)

    fig.suptitle("Primacy / Recency Rate per Difficulty Bucket", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig5_bucket_primacy.png")


# ── Figure 6: Label distribution per condition ────────────────────────────────

def fig6_label_distribution(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    cond_labels = ["Orig AB", "Orig BA", "Para AB", "Para BA"]
    label_colors = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        x      = np.arange(len(CONDITIONS))
        bottom = np.zeros(len(CONDITIONS))
        for lbl in ("A", "B", "C"):
            vals = np.array([metrics[model]["label_rates"].get(c, {}).get(lbl, 0)
                             for c in CONDITIONS])
            bars = ax.bar(x, vals, 0.55, bottom=bottom,
                          color=label_colors[lbl], edgecolor="white", label=f"Label {lbl}")
            for b, v, bot in zip(bars, vals, bottom):
                if v > 0.06:
                    ax.text(b.get_x() + b.get_width() / 2, bot + v / 2,
                            f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(cond_labels, fontsize=9)
        ax.set_ylim(0, 1); ax.set_title(model, fontsize=11, pad=6)
        ax.set_ylabel("Fraction"); ax.legend(fontsize=8)
        ax.grid(False)

    fig.suptitle("Label Distribution (A/B/C) per Condition", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig6_label_distribution.png")


# ── Figure 7: Bias decomposition (AB vs BA) ───────────────────────────────────

def fig7_bias_decomposition(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    comp_colors = {
        "position_consistency": "#55A868",
        "primacy_bias_rate":    "#4C72B0",
        "recency_bias_rate":    "#DD8452",
        "other_rate":           "#8172B2",
    }
    comp_labels = {
        "position_consistency": "Consistent",
        "primacy_bias_rate":    "Primacy Bias",
        "recency_bias_rate":    "Recency Bias",
        "other_rate":           "Other",
    }

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        x      = np.arange(2)
        bottom = np.zeros(2)
        for comp in ("position_consistency", "primacy_bias_rate", "recency_bias_rate", "other_rate"):
            vals = np.array([
                metrics[model]["position_consistency"]["original"].get(comp, 0),
                metrics[model]["position_consistency"]["para"].get(comp, 0),
            ])
            vals = np.where(np.isnan(vals), 0, vals)
            bars = ax.bar(x, vals, 0.5, bottom=bottom,
                          color=comp_colors[comp], edgecolor="white",
                          label=comp_labels[comp])
            for b, v, bot in zip(bars, vals, bottom):
                if v > 0.05:
                    ax.text(b.get_x() + b.get_width() / 2, bot + v / 2,
                            f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(["Original\n(AB vs BA)", "Para\n(AB vs BA)"])
        ax.set_ylim(0, 1); ax.set_title(model, fontsize=11, pad=6)
        ax.set_ylabel("Fraction of pairs"); ax.legend(fontsize=8)
        ax.grid(False)

    fig.suptitle("Position Bias Decomposition (AB vs BA)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig7_bias_decomposition.png")


# ── Figure 8: Volatile pairs per bucket ───────────────────────────────────────

def fig8_volatile_pairs(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = list(DIFFICULTY_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute counts
    ax = axes[0]
    x  = np.arange(len(diffs))
    w  = 0.35
    for i, model in enumerate(models):
        vol  = metrics[model].get("volatile_by_difficulty", {})
        vals = [vol.get(d, 0) for d in diffs]
        top  = max(vals) + 3
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=model, alpha=0.9)
        _ann(ax, bars, fmt=".0f", ylim=(0, top))
        ax.set_ylim(0, top)
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_xlabel("Difficulty Bucket"); ax.set_ylabel("Volatile Pair Count")
    ax.set_title("Volatile Pairs — Absolute Count", fontsize=11, pad=6)
    ax.legend(fontsize=9)

    # Right: fraction of pairs in that bucket that flipped
    ax = axes[1]
    for i, model in enumerate(models):
        vol  = metrics[model].get("volatile_by_difficulty", {})
        vals = []
        for d in diffs:
            n_bucket = metrics[model]["buckets"].get(d, {}).get("n", 0)
            n_vol    = vol.get(d, 0)
            vals.append(n_vol / n_bucket if n_bucket else 0)
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=model, alpha=0.9)
        _ann(ax, bars, ylim=(0, 0.65))
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("Difficulty Bucket"); ax.set_ylabel("Flip Rate within Bucket")
    ax.set_title("Volatile Pairs — Fraction within Bucket", fontsize=11, pad=6)
    ax.legend(fontsize=9)

    fig.suptitle("Volatile Pairs by Difficulty Bucket", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig8_volatile_pairs.png")


# ── Figure 9: Score gap vs flip rate ─────────────────────────────────────────

def fig9_score_gap_flip(dfs: dict[str, pd.DataFrame], plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in dfs]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for model in models:
        w = _wide(dfs[model])
        if "score_gap" not in w.columns:
            continue
        sub = w[["original_AB", "para_AB", "score_gap"]].dropna()
        if sub.empty:
            continue
        sub = sub.copy()
        try:
            sub["gap_bin"] = pd.qcut(sub["score_gap"], q=4,
                                     labels=["Q1\n(hardest)", "Q2", "Q3", "Q4\n(easiest)"],
                                     duplicates="drop")
        except Exception:
            # fall back to fixed bins based on difficulty thresholds
            bins = [-0.01, 0.05, 0.95, 1.62, 10]
            labels = ["tie", "hard", "medium", "easy"]
            sub["gap_bin"] = pd.cut(sub["score_gap"], bins=bins, labels=labels)

        flip_by_bin = sub.groupby("gap_bin", observed=True).apply(
            lambda g: (g["original_AB"] != g["para_AB"]).mean()
        )
        counts = sub.groupby("gap_bin", observed=True).size()

        xs = np.arange(len(flip_by_bin))
        bars = ax.bar(xs + (list(models).index(model) - 0.5) * 0.35,
                      flip_by_bin.values, 0.35,
                      color=MODEL_COLORS[model], edgecolor="white",
                      alpha=0.9, label=model)

        for b, v, cnt in zip(bars, flip_by_bin.values, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.2f}\n(n={cnt})", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(xs)
        ax.set_xticklabels(flip_by_bin.index.tolist())

    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Score Gap Quartile (score_gap = |score_a − score_b|)")
    ax.set_ylabel("Flip Rate")
    ax.set_title("Flip Rate vs Score Gap (pair difficulty proxy)", fontsize=12, pad=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, plot_dir / "fig9_score_gap_flip.png")


# ── Figure 10: Summary table ──────────────────────────────────────────────────

def fig10_summary_table(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]

    def _v(model, key_fn, fmt=".3f") -> str:
        try:
            v = key_fn(metrics[model])
            return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:{fmt}}"
        except Exception:
            return "—"

    rows = []

    def row(name, fn, fmt=".3f"):
        rows.append([name] + [_v(m, fn, fmt) for m in models])

    row("N pairs",              lambda m: m["n"],                                     fmt=".0f")
    row("Flip Rate",            lambda m: m["input_sensitivity"]["flip_rate"])
    row("Agreement",            lambda m: m["input_sensitivity"]["agreement"])
    row("Accuracy (original)",  lambda m: m["accuracy"]["original"])
    row("Accuracy (para)",      lambda m: m["accuracy"]["para"])
    row("Accuracy Δ",           lambda m: m["accuracy"]["delta"],                     fmt="+.3f")
    row("Pos. Bias (original)", lambda m: m["position_consistency"]["original"]["position_bias_rate"])
    row("Pos. Bias (para)",     lambda m: m["position_consistency"]["para"]["position_bias_rate"])
    row("Pos. Bias Δ",          lambda m: m["position_consistency"]["position_bias_delta"], fmt="+.3f")
    row("Primacy Rate (orig)",  lambda m: m["label_rates"]["original_AB"]["A"])
    row("Primacy Rate (para)",  lambda m: m["label_rates"]["para_AB"]["A"])
    row("Recency Rate (orig)",  lambda m: m["label_rates"]["original_AB"]["B"])
    row("Recency Rate (para)",  lambda m: m["label_rates"]["para_AB"]["B"])
    row("Tie Rate (orig)",      lambda m: m["label_rates"]["original_AB"]["C"])
    row("Tie Rate (para)",      lambda m: m["label_rates"]["para_AB"]["C"])

    rows.append(["─" * 28] + ["─" * 14] * len(models))

    for d in DIFFICULTY_ORDER:
        row(f"[{d}] N",            lambda m, dd=d: m["buckets"].get(dd, {}).get("n"),         fmt=".0f")
        row(f"[{d}] Flip Rate",    lambda m, dd=d: m["buckets"].get(dd, {}).get("input_sensitivity", {}).get("flip_rate"))
        row(f"[{d}] Acc Original", lambda m, dd=d: m["buckets"].get(dd, {}).get("accuracy", {}).get("original"))
        row(f"[{d}] Acc Para",     lambda m, dd=d: m["buckets"].get(dd, {}).get("accuracy", {}).get("para"))
        row(f"[{d}] Acc Δ",        lambda m, dd=d: m["buckets"].get(dd, {}).get("accuracy", {}).get("delta"), fmt="+.3f")
        row(f"[{d}] Pos. Bias",    lambda m, dd=d: m["buckets"].get(dd, {}).get("position_consistency", {}).get("original", {}).get("position_bias_rate"))

    col_labels = ["Metric"] + models
    n_rows = len(rows)

    fig, ax = plt.subplots(figsize=(10, 0.38 * n_rows + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        bg = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    fig.suptitle("B4 Paraphrase Sensitivity — Full Summary Table",
                 fontsize=12, fontweight="bold", y=0.98)
    _save(fig, plot_dir / "fig10_summary_table.png")


# ── Paraphrase quality ────────────────────────────────────────────────────────

def compute_paraphrase_quality(data_path: Path) -> pd.DataFrame | None:
    """
    Compute cosine similarity (semantic) and ROUGE-L (lexical) between each
    original response and its paraphrase.  Returns one row per response
    (response_a and response_b treated separately).
    """
    if not data_path.exists():
        print(f"  WARNING: {data_path} not found — skipping paraphrase quality")
        return None
    if not _HAVE_ST and not _HAVE_ROUGE:
        print("  WARNING: neither sentence-transformers nor rouge-score installed — skipping")
        print("  pip install sentence-transformers rouge-score")
        return None

    with open(data_path) as f:
        pairs = json.load(f)
    pairs = [p for p in pairs
             if p.get("response_a_para") and p.get("response_b_para")]
    if not pairs:
        print("  WARNING: no valid pairs in paraphrase dataset")
        return None

    # ── Semantic similarity ──
    if _HAVE_ST:
        print("  Encoding with sentence-transformers (all-MiniLM-L6-v2)...")
        st_model = _ST("all-MiniLM-L6-v2")
        orig_a = st_model.encode([p["response_a"]      for p in pairs], batch_size=64, show_progress_bar=False)
        para_a = st_model.encode([p["response_a_para"]  for p in pairs], batch_size=64, show_progress_bar=False)
        orig_b = st_model.encode([p["response_b"]      for p in pairs], batch_size=64, show_progress_bar=False)
        para_b = st_model.encode([p["response_b_para"]  for p in pairs], batch_size=64, show_progress_bar=False)

        def _cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        cos_a = [_cos(orig_a[i], para_a[i]) for i in range(len(pairs))]
        cos_b = [_cos(orig_b[i], para_b[i]) for i in range(len(pairs))]
    else:
        cos_a = cos_b = [np.nan] * len(pairs)

    # ── Lexical overlap ──
    rscorer = _rs.RougeScorer(["rougeL"], use_stemmer=True) if _HAVE_ROUGE else None

    rows = []
    for i, p in enumerate(pairs):
        for key, cos in [("a", cos_a[i]), ("b", cos_b[i])]:
            orig = p[f"response_{key}"]
            para = p[f"response_{key}_para"]
            rouge_l = (rscorer.score(orig, para)["rougeL"].fmeasure
                       if rscorer else np.nan)
            len_orig = len(orig)
            len_para = len(para)
            rows.append({
                "prompt_id":            p["prompt_id"],
                "difficulty":           p.get("difficulty"),
                "response":             key,
                "cos_sim":              cos,
                "rouge_l":              rouge_l,
                "len_orig":             len_orig,
                "len_para":             len_para,
                "len_ratio":            len_para / len_orig if len_orig else np.nan,
                "flagged_low_semantic": (cos < 0.70 if not math.isnan(cos) else False),
                "flagged_copy":         (rouge_l > 0.90 if not math.isnan(rouge_l) else False),
            })

    df = pd.DataFrame(rows)
    n_flagged = df["flagged_low_semantic"].sum() + df["flagged_copy"].sum()
    print(f"  Quality computed for {len(pairs)} pairs "
          f"({n_flagged} responses flagged out of {len(df)})")
    return df


# ── Paraphrase quality figures ────────────────────────────────────────────────

def fig_pq_distributions(qdf: pd.DataFrame, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, xlabel, good_range, copy_thresh, title in [
        (axes[0], "cos_sim",  "Cosine Similarity",  (0.85, 1.0), None,
         "Semantic Similarity Distribution"),
        (axes[1], "rouge_l",  "ROUGE-L",            (0.30, 0.70), 0.90,
         "Lexical Overlap Distribution (ROUGE-L)"),
    ]:
        for resp, color in [("a", "#4C72B0"), ("b", "#DD8452")]:
            vals = qdf[qdf["response"] == resp][col].dropna()
            ax.hist(vals, bins=25, alpha=0.6, color=color,
                    label=f"Response {resp.upper()}", edgecolor="white")
        ax.axvspan(*good_range, alpha=0.10, color="green", label="good zone")
        if col == "cos_sim":
            ax.axvline(0.70, color="red",   linestyle="--", linewidth=1.2, label="low threshold (0.7)")
            ax.axvline(0.85, color="green", linestyle="--", linewidth=1.2, label="good (0.85)")
        if copy_thresh:
            ax.axvline(copy_thresh, color="red", linestyle="--", linewidth=1.2, label="copy threshold (0.9)")
        ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        ax.set_title(title, fontsize=11, pad=6)
        ax.legend(fontsize=8)

    fig.suptitle("Paraphrase Quality — Score Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig_pq_distributions.png")


def fig_pq_scatter(qdf: pd.DataFrame, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for diff in DIFFICULTY_ORDER:
        sub = qdf[qdf["difficulty"] == diff]
        if sub.empty:
            continue
        ax.scatter(sub["rouge_l"], sub["cos_sim"],
                   color=DIFF_COLORS[diff], alpha=0.45, s=22, label=diff)

    ax.axvspan(0.30, 0.70, alpha=0.07, color="green")
    ax.axhspan(0.85, 1.00, alpha=0.07, color="green")
    ax.axhline(0.70, color="red",    linestyle="--", linewidth=1.0, alpha=0.8,
               label="low semantic threshold (0.7)")
    ax.axvline(0.90, color="orange", linestyle="--", linewidth=1.0, alpha=0.8,
               label="copy threshold (0.9 ROUGE)")
    ax.text(0.45, 0.92, "good zone", fontsize=8, color="green", alpha=0.7)

    ax.set_xlabel("ROUGE-L (lexical overlap)")
    ax.set_ylabel("Cosine Similarity (semantic)")
    ax.set_title("Paraphrase Quality: Semantic vs Lexical Similarity", fontsize=12, pad=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, plot_dir / "fig_pq_scatter.png")


def fig_pq_by_bucket(qdf: pd.DataFrame, plot_dir: Path) -> None:
    diffs = [d for d in DIFFICULTY_ORDER if d in qdf["difficulty"].values]
    x = np.arange(len(diffs))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for panel_i, (col, title, ylim, ref_lines) in enumerate([
        ("cos_sim", "Cosine Similarity per Bucket",  (0.5, 1.0),
         [(0.85, "green", "good (0.85)"), (0.70, "red", "threshold (0.7)")]),
        ("rouge_l", "ROUGE-L per Bucket",            (0.0, 1.0),
         [(0.90, "red", "copy (0.9)")]),
        (None,      "Flagged Rate per Bucket",        (0.0, 0.6), []),
    ]):
        ax = axes[panel_i]
        if col is not None:
            for resp, color in [("a", "#4C72B0"), ("b", "#DD8452")]:
                sub = qdf[qdf["response"] == resp]
                means = [sub[sub["difficulty"] == d][col].mean() for d in diffs]
                stds  = [sub[sub["difficulty"] == d][col].std()  for d in diffs]
                xi    = x + (["a", "b"].index(resp) - 0.5) * w
                bars  = ax.bar(xi, means, w, color=color, edgecolor="white",
                               alpha=0.9, label=f"Response {resp.upper()}")
                ax.errorbar(xi, means, yerr=stds, fmt="none",
                            color="black", capsize=3, linewidth=1)
                _ann(ax, bars, ylim=ylim)
            for val, color, label in ref_lines:
                ax.axhline(val, color=color, linestyle="--", linewidth=1.0,
                           alpha=0.7, label=label)
        else:
            sem  = [qdf[qdf["difficulty"] == d]["flagged_low_semantic"].mean() for d in diffs]
            copy = [qdf[qdf["difficulty"] == d]["flagged_copy"].mean()          for d in diffs]
            b1 = ax.bar(x - w/2, sem,  w, color="#d62728", edgecolor="white",
                        alpha=0.9, label="Low semantic (<0.7)")
            b2 = ax.bar(x + w/2, copy, w, color="#ff7f0e", edgecolor="white",
                        alpha=0.9, label="Near-copy (>0.9 ROUGE)")
            _ann(ax, b1, fmt=".2f", ylim=ylim)
            _ann(ax, b2, fmt=".2f", ylim=ylim)

        ax.set_xticks(x); ax.set_xticklabels(diffs)
        ax.set_ylim(*ylim); ax.set_title(title, fontsize=11, pad=6)
        ax.legend(fontsize=8)

    fig.suptitle("Paraphrase Quality per Difficulty Bucket", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig_pq_by_bucket.png")


def fig_pq_length(qdf: pd.DataFrame, plot_dir: Path) -> None:
    diffs = [d for d in DIFFICULTY_ORDER if d in qdf["difficulty"].values]
    x = np.arange(len(diffs))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for resp, color in [("a", "#4C72B0"), ("b", "#DD8452")]:
        vals = qdf[qdf["response"] == resp]["len_ratio"].dropna()
        ax.hist(vals, bins=25, alpha=0.6, color=color,
                label=f"Response {resp.upper()}", edgecolor="white")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="same length")
    ax.set_xlabel("Length Ratio (len_para / len_orig)")
    ax.set_ylabel("Count")
    ax.set_title("Length Ratio Distribution", fontsize=11, pad=6)
    ax.legend(fontsize=8)

    ax = axes[1]
    for resp, color in [("a", "#4C72B0"), ("b", "#DD8452")]:
        sub   = qdf[qdf["response"] == resp]
        means = [sub[sub["difficulty"] == d]["len_ratio"].mean() for d in diffs]
        stds  = [sub[sub["difficulty"] == d]["len_ratio"].std()  for d in diffs]
        xi    = x + (["a", "b"].index(resp) - 0.5) * w
        bars  = ax.bar(xi, means, w, color=color, edgecolor="white",
                       alpha=0.9, label=f"Response {resp.upper()}")
        ax.errorbar(xi, means, yerr=stds, fmt="none",
                    color="black", capsize=3, linewidth=1)
        _ann(ax, bars, ylim=(0.5, 1.6))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="same length")
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0.5, 1.6)
    ax.set_title("Mean Length Ratio per Bucket", fontsize=11, pad=6)
    ax.legend(fontsize=8)

    fig.suptitle("Paraphrase Length Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig_pq_length.png")


def fig_pq_table(qdf: pd.DataFrame, plot_dir: Path) -> None:
    col_labels = ["Bucket", "N", "Cos Sim mean±std", "Cos Sim min",
                  "ROUGE-L mean±std", "ROUGE-L max",
                  "Len Ratio mean±std", "% Low Semantic", "% Near-Copy"]
    rows = []

    def _row(label, sub):
        n = len(sub)
        rows.append([
            label, str(n),
            f"{sub['cos_sim'].mean():.3f} ± {sub['cos_sim'].std():.3f}",
            f"{sub['cos_sim'].min():.3f}",
            f"{sub['rouge_l'].mean():.3f} ± {sub['rouge_l'].std():.3f}",
            f"{sub['rouge_l'].max():.3f}",
            f"{sub['len_ratio'].mean():.3f} ± {sub['len_ratio'].std():.3f}",
            f"{sub['flagged_low_semantic'].mean():.1%}",
            f"{sub['flagged_copy'].mean():.1%}",
        ])

    _row("Overall", qdf)
    for d in DIFFICULTY_ORDER:
        sub = qdf[qdf["difficulty"] == d]
        if not sub.empty:
            _row(f"[{d}]", sub)
    for resp in ("a", "b"):
        sub = qdf[qdf["response"] == resp]
        if not sub.empty:
            _row(f"Response {resp.upper()}", sub)

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(16, 0.52 * n_rows + 2.0))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, n_rows + 1):
        bg = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    fig.suptitle("Paraphrase Quality — Summary Table",
                 fontsize=12, fontweight="bold", y=0.98)
    _save(fig, plot_dir / "fig_pq_table.png")


# ── Figure: Agreement per bucket ─────────────────────────────────────────────

def fig_agreement_bucket(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]
    x = np.arange(len(diffs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        vals = []
        for d in diffs:
            fr = metrics[model]["buckets"].get(d, {}).get(
                "input_sensitivity", {}).get("flip_rate", np.nan)
            vals.append(1 - fr if not math.isnan(fr) else np.nan)
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=model, alpha=0.9)
        _ann(ax, bars, ylim=(0.7, 1.0))

    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("Difficulty Bucket")
    ax.set_ylabel("Agreement (original → para)")
    ax.set_title("Input Sensitivity: Agreement per Difficulty Bucket", fontsize=12, pad=8)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4, label="perfect (1.0)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, plot_dir / "fig_agreement_bucket.png")


# ── Figure: Final summary of all metrics ─────────────────────────────────────

def fig_final_summary(metrics: dict, plot_dir: Path) -> None:
    """
    2×3 grid. Each panel with an orig/para pair shows 4 bars:
      Llama-orig | Llama-para | Apertus-orig | Apertus-para
    Panels with a single value per model show 2 bars.
    """
    models = [m for m in MODEL_ORDER if m in metrics]
    w = 0.18

    # x positions for the 4-bar layout (two groups of two, small inner gap)
    x4 = np.array([-0.28, -0.09, 0.09, 0.28])
    x2 = np.array([-0.19, 0.19])

    # (title, [(label, fn, color, hatch), ...], ylim, hline, fmt, higher_better)
    panels = [
        ("Accuracy", [
            ("Llama orig",    lambda m="Llama-3.3-70B": metrics[m]["accuracy"]["original"],    MODEL_COLORS["Llama-3.3-70B"], ""),
            ("Llama para",    lambda m="Llama-3.3-70B": metrics[m]["accuracy"]["para"],         MODEL_COLORS["Llama-3.3-70B"], "///"),
            ("Apertus orig",  lambda m="Apertus-70B":   metrics[m]["accuracy"]["original"],    MODEL_COLORS["Apertus-70B"],   ""),
            ("Apertus para",  lambda m="Apertus-70B":   metrics[m]["accuracy"]["para"],         MODEL_COLORS["Apertus-70B"],   "///"),
        ], (0.0, 0.75), None, ".3f", True),

        ("Position Consistency", [
            ("Llama orig",    lambda m="Llama-3.3-70B": metrics[m]["position_consistency"]["original"]["position_consistency"], MODEL_COLORS["Llama-3.3-70B"], ""),
            ("Llama para",    lambda m="Llama-3.3-70B": metrics[m]["position_consistency"]["para"]["position_consistency"],     MODEL_COLORS["Llama-3.3-70B"], "///"),
            ("Apertus orig",  lambda m="Apertus-70B":   metrics[m]["position_consistency"]["original"]["position_consistency"], MODEL_COLORS["Apertus-70B"],   ""),
            ("Apertus para",  lambda m="Apertus-70B":   metrics[m]["position_consistency"]["para"]["position_consistency"],     MODEL_COLORS["Apertus-70B"],   "///"),
        ], (0.0, 1.0), None, ".3f", True),

        ("Position Bias Rate", [
            ("Llama orig",    lambda m="Llama-3.3-70B": metrics[m]["position_consistency"]["original"]["position_bias_rate"], MODEL_COLORS["Llama-3.3-70B"], ""),
            ("Llama para",    lambda m="Llama-3.3-70B": metrics[m]["position_consistency"]["para"]["position_bias_rate"],     MODEL_COLORS["Llama-3.3-70B"], "///"),
            ("Apertus orig",  lambda m="Apertus-70B":   metrics[m]["position_consistency"]["original"]["position_bias_rate"], MODEL_COLORS["Apertus-70B"],   ""),
            ("Apertus para",  lambda m="Apertus-70B":   metrics[m]["position_consistency"]["para"]["position_bias_rate"],     MODEL_COLORS["Apertus-70B"],   "///"),
        ], (0.0, 0.75), None, ".3f", False),

        ("Primacy Bias Rate", [
            ("Llama orig",    lambda m="Llama-3.3-70B": metrics[m]["position_consistency"]["original"]["primacy_bias_rate"], MODEL_COLORS["Llama-3.3-70B"], ""),
            ("Llama para",    lambda m="Llama-3.3-70B": metrics[m]["position_consistency"]["para"]["primacy_bias_rate"],     MODEL_COLORS["Llama-3.3-70B"], "///"),
            ("Apertus orig",  lambda m="Apertus-70B":   metrics[m]["position_consistency"]["original"]["primacy_bias_rate"], MODEL_COLORS["Apertus-70B"],   ""),
            ("Apertus para",  lambda m="Apertus-70B":   metrics[m]["position_consistency"]["para"]["primacy_bias_rate"],     MODEL_COLORS["Apertus-70B"],   "///"),
        ], (0.0, 0.5), 1/3, ".3f", False),

        ("Agreement (orig → para)", [
            ("Llama",   lambda m="Llama-3.3-70B": metrics[m]["input_sensitivity"]["agreement"], MODEL_COLORS["Llama-3.3-70B"], ""),
            ("Apertus", lambda m="Apertus-70B":   metrics[m]["input_sensitivity"]["agreement"], MODEL_COLORS["Apertus-70B"],   ""),
        ], (0.7, 1.0), None, ".3f", True),

        ("Flip Rate (orig → para)", [
            ("Llama",   lambda m="Llama-3.3-70B": metrics[m]["input_sensitivity"]["flip_rate"], MODEL_COLORS["Llama-3.3-70B"], ""),
            ("Apertus", lambda m="Apertus-70B":   metrics[m]["input_sensitivity"]["flip_rate"], MODEL_COLORS["Apertus-70B"],   ""),
        ], (0.0, 0.35), None, ".3f", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()

    for ax, (title, bars_spec, ylim, hline, fmt, higher_better) in zip(axes_flat, panels):
        is4 = len(bars_spec) == 4
        xs  = x4 if is4 else x2

        for xi, (label, fn, color, hatch) in zip(xs, bars_spec):
            try:
                val = fn()
                val = val if not (isinstance(val, float) and math.isnan(val)) else np.nan
            except Exception:
                val = np.nan
            b = ax.bar(xi, val, w, color=color, hatch=hatch,
                       edgecolor="white", alpha=0.88, label=label)
            if not math.isnan(val):
                ax.text(xi, val + 0.012 * (ylim[1] - ylim[0]),
                        f"{val:{fmt}}", ha="center", va="bottom", fontsize=7.5)

        # x-axis labels
        if is4:
            ax.set_xticks(x4)
            ax.set_xticklabels(["Llama\norig", "Llama\npara",
                                 "Apertus\norig", "Apertus\npara"], fontsize=7.5)
            # vertical separator between the two model groups
            ax.axvline(0, color="#cccccc", linewidth=0.8, linestyle="--")
        else:
            ax.set_xticks(x2)
            ax.set_xticklabels(["Llama-3.3-70B", "Apertus-70B"], fontsize=8)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10, pad=6)
        if hline is not None:
            ax.axhline(hline, color="gray", linewidth=0.8, linestyle=":", alpha=0.6,
                       label=f"random ({hline:.2f})")
        arrow = "↑ better" if higher_better else "↓ better"
        ax.text(0.99, 0.98, arrow, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="gray")

    # Legend: model colors + orig/para hatch
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=MODEL_COLORS["Llama-3.3-70B"],  label="Llama-3.3-70B"),
        Patch(facecolor=MODEL_COLORS["Apertus-70B"],    label="Apertus-70B"),
        Patch(facecolor="#888", hatch="",   label="Original"),
        Patch(facecolor="#888", hatch="///", label="Paraphrased"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.03),
               frameon=True, edgecolor="#ccc")

    fig.suptitle("B4 Paraphrase Sensitivity — Full Metric Summary",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, plot_dir / "fig_final_summary.png")


# ── Figure: Cohen's kappa per bucket ─────────────────────────────────────────

def fig_kappa_bucket(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]
    x = np.arange(len(diffs))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: kappa per bucket
    ax = axes[0]
    for i, model in enumerate(models):
        vals = [metrics[model]["buckets"].get(d, {}).get(
                    "input_sensitivity", {}).get("kappa", np.nan)
                for d in diffs]
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=model, alpha=0.9)
        _ann(ax, bars, ylim=(-0.1, 0.6))
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(-0.1, 0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Difficulty Bucket")
    ax.set_ylabel("Cohen's κ  (original AB vs para AB)")
    ax.set_title("Cohen's κ per Difficulty Bucket", fontsize=12, pad=8)
    ax.legend(fontsize=9)

    # Right: raw agreement vs kappa side-by-side (overall)
    ax = axes[1]
    x2 = np.arange(len(models))
    for i, (key, label, hatch) in enumerate([
        ("agreement", "Raw agreement", ""),
        ("kappa",     "Cohen's κ",     "///"),
    ]):
        vals = [metrics[m]["input_sensitivity"].get(key, np.nan) for m in models]
        bars = ax.bar(x2 + (i - 0.5) * w, vals, w,
                      color=[MODEL_COLORS[m] for m in models],
                      hatch=hatch, edgecolor="white", alpha=0.9, label=label)
        _ann(ax, bars, ylim=(0.0, 1.0))
    ax.set_xticks(x2); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title("Raw Agreement vs Cohen's κ (overall)", fontsize=12, pad=8)
    ax.legend(fontsize=9)

    fig.suptitle("Bias-corrected Agreement: Cohen's κ", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig_kappa_bucket.png")


# ── Figure: Normalised sensitivity S_norm per bucket ─────────────────────────

def fig_snorm_bucket(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]
    x = np.arange(len(diffs))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: S_norm per bucket
    ax = axes[0]
    for i, model in enumerate(models):
        vals = [metrics[model]["buckets"].get(d, {}).get(
                    "input_sensitivity", {}).get("s_norm", np.nan)
                for d in diffs]
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=model, alpha=0.9)
        _ann(ax, bars, ylim=(0.0, 1.4), fmt=".2f")
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0.0, 1.4)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6,
               label="S_norm = 1  (para = positional)")
    ax.set_xlabel("Difficulty Bucket")
    ax.set_ylabel("S_norm = F_para / F_pos")
    ax.set_title("Normalised Sensitivity per Difficulty Bucket", fontsize=12, pad=8)
    ax.legend(fontsize=9)

    # Right: F_para, F_pos, S_norm overall — grouped bar
    ax = axes[1]
    x2 = np.arange(len(models))
    specs = [
        ("F_pos  (position flip)",  "f_pos",  "///", 0.6),
        ("F_para (paraphrase flip)", "flip_rate", "",  0.9),
    ]
    for i, (label, key, hatch, alpha) in enumerate(specs):
        vals = [metrics[m]["input_sensitivity"].get(key, np.nan) for m in models]
        bars = ax.bar(x2 + (i - 0.5) * w, vals, w,
                      color=[MODEL_COLORS[m] for m in models],
                      hatch=hatch, edgecolor="white", alpha=alpha, label=label)
        _ann(ax, bars, ylim=(0.0, 0.55), fmt=".3f")

    # S_norm as text annotation above each model group
    for j, model in enumerate(models):
        sn = metrics[model]["input_sensitivity"].get("s_norm", np.nan)
        if not math.isnan(sn):
            ax.text(x2[j], 0.51, f"S={sn:.2f}",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=MODEL_COLORS[model])

    ax.set_xticks(x2); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.0, 0.55)
    ax.set_title("F_pos vs F_para (overall) with S_norm", fontsize=12, pad=8)
    ax.legend(fontsize=9)

    fig.suptitle("Normalised Sensitivity  S_norm = F_para / F_pos",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig_snorm_bucket.png")


# ── Figure: Agreement + κ per bucket — only position-consistent pairs ──────────

def fig_agreement_kappa_consistent(dfs: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in dfs]
    diffs  = DIFFICULTY_ORDER
    x = np.arange(len(diffs))
    bar_w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # pre-compute consistent wide dfs
    wcs = {}
    for model in models:
        w = _wide(dfs[model])
        if "original_AB" not in w.columns or "original_BA" not in w.columns:
            continue
        mask = w.apply(lambda r: FLIP_MAP.get(r["original_AB"]) == r["original_BA"], axis=1)
        wcs[model] = w[mask]

    # Left: agreement per bucket
    ax = axes[0]
    for i, model in enumerate(models):
        wc = wcs.get(model)
        if wc is None:
            continue
        vals = []
        for d in diffs:
            sub = wc[wc["difficulty"] == d][["original_AB", "para_AB"]].dropna()
            vals.append(float((sub["original_AB"] == sub["para_AB"]).mean()) if len(sub) else np.nan)
        bars = ax.bar(x + (i - 0.5) * bar_w, vals, bar_w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=f"{model} (n={len(wcs[model])})", alpha=0.9)
        _ann(ax, bars, ylim=(0.7, 1.0))
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("Difficulty Bucket")
    ax.set_ylabel("Agreement (original → para)")
    ax.set_title("Paraphrase Agreement", fontsize=12, pad=8)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    # Right: Cohen's κ per bucket
    ax = axes[1]
    for i, model in enumerate(models):
        wc = wcs.get(model)
        if wc is None:
            continue
        vals = []
        for d in diffs:
            sub = wc[wc["difficulty"] == d]
            vals.append(_cohens_kappa(sub, "original_AB", "para_AB"))
        bars = ax.bar(x + (i - 0.5) * bar_w, vals, bar_w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=f"{model} (n={len(wcs[model])})", alpha=0.9)
        _ann(ax, bars, ylim=(0.0, 1.0))
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Difficulty Bucket")
    ax.set_ylabel("Cohen's κ  (original AB vs para AB)")
    ax.set_title("Bias-corrected Agreement (Cohen's κ)", fontsize=12, pad=8)
    ax.legend(fontsize=9)

    fig.suptitle("Paraphrase Robustness on Position-Consistent Pairs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig_agreement_kappa_consistent.png")


# ── Figure: S_norm (left) + Cohen's κ (right) combined ───────────────────────

def fig_kappa_snorm_combined(metrics: dict, plot_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if m in metrics]
    diffs  = [d for d in DIFFICULTY_ORDER
              if any(d in metrics[m].get("buckets", {}) for m in models)]
    x = np.arange(len(diffs))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: S_norm per bucket
    ax = axes[0]
    for i, model in enumerate(models):
        vals = [metrics[model]["buckets"].get(d, {}).get(
                    "input_sensitivity", {}).get("s_norm", np.nan)
                for d in diffs]
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=model, alpha=0.9)
        _ann(ax, bars, ylim=(0.0, 1.0), fmt=".2f")
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Difficulty Bucket")
    ax.set_ylabel("S_norm = F_para / F_pos")
    ax.set_title("Normalised Sensitivity (S_norm) per Bucket", fontsize=12, pad=8)
    ax.legend(fontsize=9)

    # Right: Cohen's κ per bucket
    ax = axes[1]
    for i, model in enumerate(models):
        vals = [metrics[model]["buckets"].get(d, {}).get(
                    "input_sensitivity", {}).get("kappa", np.nan)
                for d in diffs]
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      color=MODEL_COLORS[model], edgecolor="white",
                      label=model, alpha=0.9)
        _ann(ax, bars, ylim=(0.0, 1.0))
    ax.set_xticks(x); ax.set_xticklabels(diffs)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Difficulty Bucket")
    ax.set_ylabel("Cohen's κ  (original AB vs para AB)")
    ax.set_title("Bias-corrected Agreement (Cohen's κ) per Bucket", fontsize=12, pad=8)
    ax.legend(fontsize=9)

    fig.suptitle("Paraphrase Robustness: S_norm and Cohen's κ by Difficulty",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, plot_dir / "fig_kappa_snorm_combined.png")


# ── JSON serialisation ────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _bad_prompt_ids(qdf: pd.DataFrame, min_cosine: float) -> set[str]:
    """Return prompt_ids where any response has cosine similarity below threshold."""
    flagged = qdf[qdf["cos_sim"] < min_cosine]["prompt_id"]
    return set(flagged)


def main(args) -> None:
    results_dir = Path(args.results_dir)
    plot_dir    = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    dfs = load_all(results_dir)
    if not dfs:
        raise FileNotFoundError(f"No .jsonl files found in {results_dir}")

    # ── Paraphrase quality (computed first so we can filter) ──
    data_path = Path(args.data_path) if args.data_path else None
    if data_path is None:
        split = args.split
        candidate = Path("data") / f"helpsteer2_{split}_paraphrased.json"
        if candidate.exists():
            data_path = candidate

    qdf = None
    if data_path is not None:
        print(f"\nComputing paraphrase quality from {data_path}...")
        qdf = compute_paraphrase_quality(data_path)
        if qdf is not None:
            bad_ids = _bad_prompt_ids(qdf, args.min_cosine)
            if bad_ids:
                print(f"  Filtering {len(bad_ids)} pairs with cosine sim < {args.min_cosine}")
                dfs = {name: df[~df["prompt_id"].isin(bad_ids)].copy()
                       for name, df in dfs.items()}
            else:
                print(f"  No pairs filtered (all cosine sim ≥ {args.min_cosine})")
    else:
        print("\nNo paraphrase dataset found — skipping quality filter and quality plots.")
        print("Pass --data-path data/helpsteer2_train_paraphrased.json to enable.")

    print("\nComputing metrics (overall + per-bucket)...")
    all_metrics = {name: compute_all_metrics(df) for name, df in dfs.items()}

    out_path = results_dir / "full_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_default)
    print(f"  → full_metrics.json")

    print("Generating plots...")
    fig1_overview(all_metrics, plot_dir)
    fig1b_bucket_overview(all_metrics, plot_dir)
    fig2_bucket_flip(all_metrics, plot_dir)
    fig3_bucket_accuracy(all_metrics, plot_dir)
    fig4_bucket_position(all_metrics, plot_dir)
    fig5_bucket_primacy(all_metrics, plot_dir)
    fig6_label_distribution(all_metrics, plot_dir)
    fig7_bias_decomposition(all_metrics, plot_dir)
    fig8_volatile_pairs(all_metrics, plot_dir)
    fig9_score_gap_flip(dfs, plot_dir)
    fig10_summary_table(all_metrics, plot_dir)
    fig_agreement_bucket(all_metrics, plot_dir)
    fig_final_summary(all_metrics, plot_dir)
    fig_kappa_bucket(all_metrics, plot_dir)
    fig_snorm_bucket(all_metrics, plot_dir)
    fig_kappa_snorm_combined(all_metrics, plot_dir)
    fig_agreement_kappa_consistent(dfs, plot_dir)

    if qdf is not None:
        fig_pq_distributions(qdf, plot_dir)
        fig_pq_scatter(qdf, plot_dir)
        fig_pq_by_bucket(qdf, plot_dir)
        fig_pq_length(qdf, plot_dir)
        fig_pq_table(qdf, plot_dir)
        qdf.to_csv(plot_dir / "paraphrase_quality.csv", index=False)
        print("  → paraphrase_quality.csv")

    saved = len(list(plot_dir.glob("*.png")))
    print(f"\nDone — {saved} plots saved to {plot_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/b4_paraphrase_sensitivity")
    p.add_argument("--split",       default="validation",
                   help="Dataset split used to auto-detect the paraphrased JSON")
    p.add_argument("--data-path",   default=None,
                   help="Explicit path to helpsteer2_*_paraphrased.json")
    p.add_argument("--min-cosine",  type=float, default=0.85,
                   help="Filter pairs whose cosine sim drops below this (default: 0.85)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
