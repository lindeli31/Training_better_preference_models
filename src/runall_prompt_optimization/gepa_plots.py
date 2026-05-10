"""
gepa_plots.py
-------------
Visualise GEPA training dynamics and final metrics.

Two figures are produced in `<output_dir>/plots/`:

  1. gepa_trajectory.png
     Three panels (consistency, accuracy, loss) showing a ROLLING-MEAN curve
     over the GEPA metric-call index. Each point = mean over the last W
     calls (W defaults to 10 or len(history)//20, whichever is larger), so
     the curve is well-defined even when GEPA accepts only a handful of
     distinct candidate prompts. Train and val curves are drawn separately
     using the `split` field recorded by gepa_metric().

  2. gepa_comparison.png
     Final bar chart: baseline vs GEPA-optimised judge on train + val
     pairs, using the project's compute_position_bias. This is the clean
     "before vs after" view, with a real train/val split (unlike the
     trajectory which mixes train + val evaluations GEPA performs
     internally).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Simple causal rolling mean over `values` with size `window`.
    Returns a list of the same length. The first `window-1` outputs use a
    shrinking prefix so the curve starts at x=0 (not at x=window)."""
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        denom = min(i + 1, window)
        out.append(acc / denom)
    return out


def _split_series(history: list[dict], split: str, key: str):
    """Extract per-call (x, y) for one split, where x = global call index in
    history (so train and val share the same x-axis). Points where split
    doesn't match are skipped — giving a dotted-trail-style curve."""
    xs, ys = [], []
    for i, e in enumerate(history):
        if e.get("split") != split:
            continue
        if key == "consistency":
            ys.append(float(e["consistent"]))
        elif key == "accuracy":
            ys.append(float(e["accurate"]))
        elif key == "loss":
            ys.append(1.0 - float(e["score"]))
        else:
            continue
        xs.append(i)
    return xs, ys


def plot_training_trajectory(history: list[dict], output_dir: Path) -> None:
    """Three panels (consistency, accuracy, loss) vs GEPA metric-call index.

    Each panel plots a rolling mean per split (train / val), so the curves
    are smooth even when GEPA only accepts a handful of distinct candidate
    prompts. The raw per-call 0/1 signal is drawn faintly behind the smoothed
    line to expose variance.
    """
    if not history:
        return  # nothing to plot

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Window scales with history size, but with a sensible floor so that for
    # ~100-call runs we still get a readable smoothed curve.
    window = max(10, len(history) // 20)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ("consistency", "Position consistency over GEPA calls",
         "Consistency", (0, 1.05)),
        ("accuracy",    "Accuracy over GEPA calls",
         "Accuracy",    (0, 1.05)),
        ("loss",        "GEPA loss over calls  (1 − score)",
         "Loss",        None),
    ]
    colors = {"train": "tab:blue", "val": "tab:orange"}
    markers = {"train": "o", "val": "s"}

    for ax, (key, title, ylabel, ylim) in zip(axes, panels):
        for split in ("train", "val"):
            xs, ys = _split_series(history, split, key)
            if not xs:
                continue
            smoothed = _rolling_mean(ys, window)
            ax.plot(xs, ys, marker=markers[split], linestyle="",
                    color=colors[split], alpha=0.15, markersize=3)
            ax.plot(xs, smoothed, color=colors[split], linewidth=2,
                    label=f"{split} (rolling mean, w={window})")
        ax.set_title(title)
        ax.set_xlabel("GEPA metric-call index")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "gepa_trajectory.png", dpi=150)
    plt.close(fig)


def plot_baseline_vs_optimised(
    baseline_train: dict,
    baseline_val: dict,
    train: dict,
    val: dict,
    output_dir: Path,
) -> None:
    """Bar chart comparing baseline vs GEPA-optimised judge on train & val.

    Shows four bars per metric so the reader can read both the absolute
    level and the GEPA lift on each split independently.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    labels = ["position_consistency", "position_bias_rate", "accuracy"]
    bt = [baseline_train.get(k, 0.0) for k in labels]
    bv = [baseline_val.get(k, 0.0)   for k in labels]
    tr = [train.get(k, 0.0)          for k in labels]
    vl = [val.get(k, 0.0)            for k in labels]

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 1.5 * w, bt, w, label="Baseline (train)")
    ax.bar(x - 0.5 * w, bv, w, label="Baseline (val)")
    ax.bar(x + 0.5 * w, tr, w, label="GEPA (train)")
    ax.bar(x + 1.5 * w, vl, w, label="GEPA (val)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("position_", "") for l in labels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Baseline vs GEPA-optimised judge")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "gepa_comparison.png", dpi=150)
    plt.close(fig)
