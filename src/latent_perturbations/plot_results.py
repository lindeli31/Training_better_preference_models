"""
plot_results.py
---------------
Visualize gradient probe experiment results.

Usage:
    python -m src.latent_perturbations.plot_results --results-dir results/gradient_probe_14b_v2_2074118
    python -m src.latent_perturbations.plot_results --results-dir results/gradient_probe_14b_v2_2074118 --out-dir figures/gradient_probe
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── colour palette ────────────────────────────────────────────────────────────
C_GRAD  = "#2563EB"   # blue   — gradient direction
C_RAND  = "#DC2626"   # red    — random baseline
C_BIAS  = "#7C3AED"   # purple — biased pairs
C_UNBIAS = "#059669"  # green  — unbiased pairs
C_NORM  = "#F59E0B"   # amber  — gradient norm


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Experiment 1: layer sweep ─────────────────────────────────────────────────

def plot_exp1(results_dir: Path, out_dir: Path, model_tag: str):
    summary_path = results_dir / "exp1_summary.json"
    sweep_path   = results_dir / "exp1_layer_sweep.jsonl"

    if not summary_path.exists() and not sweep_path.exists():
        print("  [skip] exp1 files not found")
        return

    # Prefer summary; fall back to raw JSONL
    if summary_path.exists():
        s = _load_json(summary_path)
        if "layers" in s:
            # Full-run summary: {layers: [...], flip_rates: [...], grad_norms: [...]}
            layers      = s["layers"]
            flip_rates  = s["flip_rates"]
            grad_norms  = s.get("grad_norms")
        else:
            # Smoke-test summary: {"0": {lasr, mean_grad_norm, n_pairs}, ...}
            layers     = sorted(int(k) for k in s)
            flip_rates = [s[str(l)]["lasr"] for l in layers]
            grad_norms = [s[str(l)].get("mean_grad_norm") for l in layers]
    else:
        # Raw per-pair JSONL: aggregate flip rate and grad norm per layer
        records  = _load_jsonl(sweep_path)
        layer_map: dict[int, list] = {}
        for r in records:
            li = r.get("layer_idx", r.get("layer"))
            layer_map.setdefault(li, []).append(r)
        layers     = sorted(layer_map)
        flip_rates = [float(np.mean([r["flipped"] for r in layer_map[l]])) for l in layers]
        grad_norms = [float(np.mean([r["grad_norm"] for r in layer_map[l] if r.get("grad_norm") is not None]))
                      if any(r.get("grad_norm") is not None for r in layer_map[l]) else None
                      for l in layers]

    fig, ax1 = plt.subplots(figsize=(9, 4))

    ax1.plot(layers, flip_rates, "o-", color=C_GRAD, linewidth=2, markersize=5, label="LASR (flip rate)")
    ax1.set_xlabel("Layer index", fontsize=12)
    ax1.set_ylabel("LASR (flip rate)", color=C_GRAD, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=C_GRAD)
    ax1.set_ylim(-0.02, max(flip_rates) * 1.25 + 0.02)

    if grad_norms and any(v is not None for v in grad_norms):
        ax2 = ax1.twinx()
        valid = [(l, g) for l, g in zip(layers, grad_norms) if g is not None]
        lv, gv = zip(*valid)
        ax2.plot(lv, gv, "s--", color=C_NORM, linewidth=1.5, markersize=4, alpha=0.8, label="Gradient norm")
        ax2.set_ylabel("Mean gradient norm", color=C_NORM, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=C_NORM)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    else:
        ax1.legend(fontsize=10)

    peak_layer = layers[int(np.argmax(flip_rates))]
    ax1.axvline(peak_layer, color=C_GRAD, linestyle=":", alpha=0.5)
    ax1.text(peak_layer + 0.3, max(flip_rates) * 1.05, f"peak L{peak_layer}", color=C_GRAD, fontsize=9)

    ax1.set_title(f"Exp 1 — Layer-wise attack success rate  |  {model_tag}", fontsize=13)
    ax1.grid(True, alpha=0.3)

    out = out_dir / "exp1_layer_sweep.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Experiment 2: antisymmetry summary ───────────────────────────────────────

def plot_exp2(results_dir: Path, out_dir: Path, model_tag: str):
    summary_path = results_dir / "exp2_summary.json"
    detail_path  = results_dir / "exp2_antisymmetry.jsonl"

    if not summary_path.exists() and not detail_path.exists():
        print("  [skip] exp2 files not found")
        return

    records = _load_jsonl(detail_path) if detail_path.exists() else []
    cosines = [r["cosine"] for r in records if r.get("cosine") is not None and not math.isnan(r["cosine"])]

    if not cosines:
        print("  [skip] no valid cosine values in exp2")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cosines, bins=30, color=C_GRAD, edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(cosines), color="black", linestyle="--", linewidth=1.5, label=f"mean={np.mean(cosines):.3f}")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("cosine( g_AB , −g_BA )", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Exp 2 — Swap antisymmetry distribution  |  {model_tag}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = out_dir / "exp2_antisymmetry.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Experiment 3: biased vs unbiased cosine ───────────────────────────────────

def plot_exp3(results_dir: Path, out_dir: Path, model_tag: str):
    classifier_path = results_dir / "exp3_classifier.jsonl"
    summary_path    = results_dir / "exp3_summary.json"

    if not classifier_path.exists() and not summary_path.exists():
        print("  [skip] exp3 files not found")
        return

    if classifier_path.exists():
        records  = _load_jsonl(classifier_path)
        biased   = [r["cosine"] for r in records if r.get("biased") is True  and r.get("cosine") is not None and not math.isnan(r["cosine"])]
        unbiased = [r["cosine"] for r in records if r.get("biased") is False and r.get("cosine") is not None and not math.isnan(r["cosine"])]
    else:
        s = _load_json(summary_path)
        # fall back to summary stats only
        biased   = None
        unbiased = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: overlapping histograms
    ax = axes[0]
    if biased and unbiased:
        bins = np.linspace(-1, 1, 35)
        ax.hist(biased,   bins=bins, color=C_BIAS,   alpha=0.65, label=f"Biased (n={len(biased)})",   edgecolor="white")
        ax.hist(unbiased, bins=bins, color=C_UNBIAS, alpha=0.65, label=f"Unbiased (n={len(unbiased)})", edgecolor="white")
        ax.axvline(np.mean(biased),   color=C_BIAS,   linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(unbiased), color=C_UNBIAS, linestyle="--", linewidth=1.5)
        ax.set_xlabel("cosine( g_AB , −g_BA )", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Antisymmetry by bias label", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Right: mean ± std bar chart
    ax2 = axes[1]
    if biased and unbiased:
        means = [np.mean(biased), np.mean(unbiased)]
        stds  = [np.std(biased),  np.std(unbiased)]
        labels = ["Biased", "Unbiased"]
        colors = [C_BIAS, C_UNBIAS]
        bars = ax2.bar(labels, means, yerr=stds, color=colors, alpha=0.8,
                       capsize=6, edgecolor="white", width=0.4)
        ax2.axhline(0, color="grey", linestyle=":", linewidth=1)
        ax2.set_ylabel("Mean cosine ± std", fontsize=12)
        ax2.set_title("Mean antisymmetry score", fontsize=12)
        # Cohen's d annotation
        pooled_std = math.sqrt((np.std(biased)**2 + np.std(unbiased)**2) / 2)
        d = (np.mean(biased) - np.mean(unbiased)) / (pooled_std + 1e-12)
        ax2.text(0.5, 0.92, f"Cohen's d = {d:.2f}", transform=ax2.transAxes,
                 ha="center", fontsize=11, color="black",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"))
        ax2.grid(True, alpha=0.3, axis="y")
    elif summary_path.exists():
        s = _load_json(summary_path)
        ax2.text(0.5, 0.5, json.dumps(s, indent=2), transform=ax2.transAxes,
                 ha="center", va="center", fontsize=8, family="monospace")

    fig.suptitle(f"Exp 3 — Classifier: biased vs unbiased  |  {model_tag}", fontsize=13)
    out = out_dir / "exp3_classifier.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Experiment 4: dose-response ───────────────────────────────────────────────

def plot_exp4(results_dir: Path, out_dir: Path, model_tag: str):
    summary_path = results_dir / "exp4_summary.json"
    detail_path  = results_dir / "exp4_dose_response.jsonl"

    if not summary_path.exists() and not detail_path.exists():
        print("  [skip] exp4 files not found")
        return

    records = None
    if summary_path.exists():
        s = _load_json(summary_path)
        if isinstance(s, list):
            records = s
        elif "alphas" in s:
            alphas         = s["alphas"]
            grad_flips     = s["grad_flip_rates"]
            rand_flip_mean = s.get("rand_flip_rate_mean")
            rand_flip_std  = s.get("rand_flip_rate_std")
        else:
            records = list(s.values())
    if records is None and detail_path.exists():
        records = _load_jsonl(detail_path)
    if records is not None:
        # Aggregate per alpha across pair-level or already-aggregated records
        alpha_map: dict[float, list] = {}
        for r in records:
            alpha_map.setdefault(r["alpha"], []).append(r)
        alphas         = sorted(alpha_map)
        grad_flips     = [float(np.mean([r.get("grad_flip_rate", r.get("flipped", 0)) for r in alpha_map[a]])) for a in alphas]
        rand_flip_mean = [float(np.mean([r.get("rand_flip_rate_mean", r.get("rand_flip_mean", 0)) for r in alpha_map[a]])) for a in alphas]
        rand_flip_std  = [float(np.mean([r.get("rand_flip_rate_std", 0) for r in alpha_map[a]])) for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(alphas, grad_flips, "o-", color=C_GRAD, linewidth=2.5, markersize=6, label="Gradient direction (LASR)")

    if rand_flip_mean is not None:
        if isinstance(rand_flip_mean, list):
            rm = rand_flip_mean
            rs = rand_flip_std if isinstance(rand_flip_std, list) else [0] * len(rm)
        else:
            rm = [rand_flip_mean] * len(alphas)
            rs = [rand_flip_std or 0] * len(alphas)
        rm = np.array(rm, dtype=float)
        rs = np.array(rs, dtype=float)
        ax.plot(alphas, rm, "s--", color=C_RAND, linewidth=2, markersize=5, label="Random direction (baseline)")
        ax.fill_between(alphas, rm - rs, rm + rs, color=C_RAND, alpha=0.15)

    ax.set_xlabel("Perturbation scale α", fontsize=12)
    ax.set_ylabel("Flip rate", fontsize=12)
    ax.set_title(f"Exp 4 — Dose-response: gradient vs random steering  |  {model_tag}", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = out_dir / "exp4_dose_response.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Combined summary panel ────────────────────────────────────────────────────

def plot_summary_panel(results_dir: Path, out_dir: Path, model_tag: str):
    """4-panel figure combining all experiments."""
    paths = {
        "exp1_summary":    results_dir / "exp1_summary.json",
        "exp1_sweep":      results_dir / "exp1_layer_sweep.jsonl",
        "exp2_detail":     results_dir / "exp2_antisymmetry.jsonl",
        "exp3_classifier": results_dir / "exp3_classifier.jsonl",
        "exp4_summary":    results_dir / "exp4_summary.json",
    }

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"Gradient Probe — Positional Bias in LLM Judge\n{model_tag}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    # ── panel 0: layer sweep ──
    ax = axes[0]
    loaded = False
    if paths["exp1_summary"].exists():
        s = _load_json(paths["exp1_summary"])
        if "layers" in s:
            layers, flip_rates = s["layers"], s["flip_rates"]
            grad_norms = s.get("grad_norms")
        else:
            layers     = sorted(int(k) for k in s)
            flip_rates = [s[str(l)]["lasr"] for l in layers]
            grad_norms = [s[str(l)].get("mean_grad_norm") for l in layers]
        loaded = True
    elif paths["exp1_sweep"].exists():
        records  = _load_jsonl(paths["exp1_sweep"])
        layer_map: dict[int, list] = {}
        for r in records:
            li = r.get("layer_idx", r.get("layer"))
            layer_map.setdefault(li, []).append(r)
        layers     = sorted(layer_map)
        flip_rates = [float(np.mean([r["flipped"] for r in layer_map[l]])) for l in layers]
        grad_norms = [float(np.mean([r["grad_norm"] for r in layer_map[l] if r.get("grad_norm") is not None]))
                      if any(r.get("grad_norm") is not None for r in layer_map[l]) else None
                      for l in layers]
        loaded = True
    if loaded:
        ax.plot(layers, flip_rates, "o-", color=C_GRAD, linewidth=2, markersize=4)
        peak_layer = layers[int(np.argmax(flip_rates))]
        ax.axvline(peak_layer, color=C_GRAD, linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer", fontsize=10); ax.set_ylabel("LASR", color=C_GRAD, fontsize=10)
        if grad_norms and any(v is not None for v in grad_norms):
            ax2 = ax.twinx()
            valid = [(l, g) for l, g in zip(layers, grad_norms) if g is not None]
            lv, gv = zip(*valid)
            ax2.plot(lv, gv, "s--", color=C_NORM, linewidth=1.2, markersize=3, alpha=0.8)
            ax2.set_ylabel("Grad norm", color=C_NORM, fontsize=9)
            ax2.tick_params(axis="y", labelcolor=C_NORM, labelsize=8)
        ax.set_title("Exp 1 — Layer-wise LASR", fontsize=11)
        ax.grid(True, alpha=0.25)

    # ── panel 1: antisymmetry distribution ──
    ax = axes[1]
    if paths["exp2_detail"].exists():
        records = _load_jsonl(paths["exp2_detail"])
        cosines = [r["cosine"] for r in records if r.get("cosine") is not None and not math.isnan(r["cosine"])]
        if cosines:
            ax.hist(cosines, bins=25, color=C_GRAD, edgecolor="white", alpha=0.8)
            ax.axvline(np.mean(cosines), color="black", linestyle="--", linewidth=1.5, label=f"μ={np.mean(cosines):.3f}")
            ax.axvline(0, color="grey", linestyle=":", linewidth=1)
            ax.set_xlabel("cosine( g_AB , −g_BA )", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title("Exp 2 — Swap antisymmetry", fontsize=11)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # ── panel 2: biased vs unbiased ──
    ax = axes[2]
    if paths["exp3_classifier"].exists():
        records  = _load_jsonl(paths["exp3_classifier"])
        biased   = [r["cosine"] for r in records if r.get("biased") is True  and r.get("cosine") is not None and not math.isnan(r["cosine"])]
        unbiased = [r["cosine"] for r in records if r.get("biased") is False and r.get("cosine") is not None and not math.isnan(r["cosine"])]
        if biased and unbiased:
            bins = np.linspace(-1, 1, 28)
            ax.hist(biased,   bins=bins, color=C_BIAS,   alpha=0.65, label=f"Biased (n={len(biased)})",   edgecolor="white")
            ax.hist(unbiased, bins=bins, color=C_UNBIAS, alpha=0.65, label=f"Unbiased (n={len(unbiased)})", edgecolor="white")
            ax.axvline(np.mean(biased),   color=C_BIAS,   linestyle="--", linewidth=1.5)
            ax.axvline(np.mean(unbiased), color=C_UNBIAS, linestyle="--", linewidth=1.5)
            pooled_std = math.sqrt((np.std(biased)**2 + np.std(unbiased)**2) / 2)
            d = (np.mean(biased) - np.mean(unbiased)) / (pooled_std + 1e-12)
            ax.set_xlabel("cosine( g_AB , −g_BA )", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"Exp 3 — Biased vs Unbiased  (d={d:.2f})", fontsize=11)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # ── panel 3: dose-response ──
    ax = axes[3]
    if paths["exp4_summary"].exists():
        s = _load_json(paths["exp4_summary"])
        if isinstance(s, list):
            recs4 = s
        elif "alphas" in s:
            recs4 = None
            alphas = s["alphas"]
            grad_flips = s["grad_flip_rates"]
            rm = s.get("rand_flip_rate_mean")
            rs = s.get("rand_flip_rate_std")
        else:
            recs4 = list(s.values())
        if isinstance(s, list) or (not isinstance(s, list) and "alphas" not in s):
            alpha_map4: dict[float, list] = {}
            for r in recs4:
                alpha_map4.setdefault(r["alpha"], []).append(r)
            alphas = sorted(alpha_map4)
            grad_flips = [float(np.mean([r.get("grad_flip_rate", r.get("flipped", 0)) for r in alpha_map4[a]])) for a in alphas]
            rm = [float(np.mean([r.get("rand_flip_rate_mean", 0) for r in alpha_map4[a]])) for a in alphas]
            rs = [float(np.mean([r.get("rand_flip_rate_std", 0) for r in alpha_map4[a]])) for a in alphas]
        ax.plot(alphas, grad_flips, "o-", color=C_GRAD, linewidth=2, markersize=5, label="Gradient")
        if rm is not None:
            if isinstance(rm, list):
                rm_arr = np.array(rm, dtype=float)
                rs_arr = np.array(rs, dtype=float) if isinstance(rs, list) else np.zeros_like(rm_arr)
            else:
                rm_arr = np.full(len(alphas), rm)
                rs_arr = np.full(len(alphas), rs or 0)
            ax.plot(alphas, rm_arr, "s--", color=C_RAND, linewidth=1.5, markersize=4, label="Random")
            ax.fill_between(alphas, rm_arr - rs_arr, rm_arr + rs_arr, color=C_RAND, alpha=0.15)
        ax.set_xlabel("α", fontsize=10); ax.set_ylabel("Flip rate", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Exp 4 — Dose-response", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    out = out_dir / "summary_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory with experiment JSON/JSONL files")
    parser.add_argument("--out-dir", default=None, help="Output directory for figures (default: results_dir/figures)")
    parser.add_argument("--model-tag", default=None, help="Model label for plot titles (inferred from dir name if omitted)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model_tag or results_dir.name

    print(f"Plotting from: {results_dir}")
    print(f"Output to:     {out_dir}")

    plot_exp1(results_dir, out_dir, model_tag)
    plot_exp2(results_dir, out_dir, model_tag)
    plot_exp3(results_dir, out_dir, model_tag)
    plot_exp4(results_dir, out_dir, model_tag)
    plot_summary_panel(results_dir, out_dir, model_tag)


if __name__ == "__main__":
    main()
