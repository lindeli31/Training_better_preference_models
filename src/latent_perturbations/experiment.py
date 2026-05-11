"""
experiment.py
-------------
Gradient-based positional bias probe — four experiments.

Exp 1 — Layer sweep (LASR-style)
    Compute ∂NLL(second_position_token)/∂h_l for each layer l and perturb
    by alpha * g/||g||. LASR(l) = flip rate across pairs. Identifies which
    layer is most sensitive to positional bias.

Exp 2 — Swap antisymmetry
    cosine( g_AB , -g_BA ) at the target layer.
    ≈1.0 → gradient is a pure position signal; ≈0.0 → content-driven.

Exp 3 — Antisymmetry as positional bias classifier
    Label pairs as biased (verdict_AB == verdict_BA) vs unbiased.
    Test whether antisymmetry score predicts the bias label (ROC).

Exp 4 — Dose-response at target layer
    Vary alpha in [0.1 … 2.0], measure flip rate and ΔNLL for gradient
    direction vs random orthogonal baseline.

Usage
-----
    python -m src.latent_perturbations.experiment --experiment 1
    python -m src.latent_perturbations.experiment --experiment all \\
        --model Qwen/Qwen2.5-32B-Instruct --target-layer 32 --n-pairs 200
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from src.latent_perturbations.config import GradientProbeConfig
from src.latent_perturbations.dataset import load_swap_pairs
from src.latent_perturbations.model import (
    load_model,
    resolve_layer_indices,
    compute_layer_outputs,
    get_verdict,
    get_verdict_with_gradient_steer,
)
from src.latent_perturbations.gradient_probe import (
    compute_swap_antisymmetry,
    classify_positional_bias,
    random_direction_baseline,
)


def _out(config: GradientProbeConfig) -> Path:
    p = Path(config.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clear_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Experiment 1: Layer sweep — LASR
# ---------------------------------------------------------------------------

def run_experiment_1(
    config: GradientProbeConfig, model, tokenizer, swap_pairs,
    sanity_check: bool = False,
) -> int:
    """
    For each layer l and each pair, steer by alpha * g/||g|| and record flip.
    LASR(l) = mean flip rate across pairs.

    sanity_check=True additionally runs the paper's sign convention:
      h + α · ∇NLL(first_pos_token)   [makes current output less likely]
    alongside our default:
      h - α · ∇NLL(second_pos_token)  [makes target token more likely]
    Both should cause the same flip direction if the gradient is a clean
    position signal.  Results stored in lasr_alt per layer.

    Returns the peak layer index (highest LASR from the default direction).
    """
    layer_indices = resolve_layer_indices(model, config)
    out = _out(config)

    print(f"\n=== Experiment 1: Layer Sweep ===")
    print(f"  layers: {layer_indices}")
    print(f"  pairs:  {len(swap_pairs)}, alpha: {config.alpha}")
    if sanity_check:
        print(f"  sanity_check=True: also running h + α·∇NLL(first_pos_token)")
    if config.normalize:
        print(f"  normalize=True: applying LatentSafety distribution normalization")

    lasr_per_layer = {}

    with open(out / "exp1_layer_sweep.jsonl", "w") as f:
        for layer_idx in tqdm(layer_indices, desc="layers"):
            flips, flips_alt, grad_norms = [], [], []

            for pair in tqdm(swap_pairs, desc=f"  layer={layer_idx}", leave=False):
                try:
                    # --- default: h - α · ∇NLL(second_pos_token) ---
                    data = compute_layer_outputs(
                        model, tokenizer,
                        pair.prompt_ab,
                        pair.second_token_ab,
                        [layer_idx],
                    )
                    if layer_idx not in data or data[layer_idx]["gradient"] is None:
                        continue

                    act      = data[layer_idx]["activation"]
                    grad_b   = data[layer_idx]["gradient"]
                    nll_orig = data[layer_idx]["nll"]

                    v_orig = get_verdict(model, tokenizer, pair.prompt_ab,
                                        config.verdict_tokens)
                    v_steered, _ = get_verdict_with_gradient_steer(
                        model, tokenizer, pair.prompt_ab,
                        layer_idx, act, grad_b, config.alpha,
                        config.verdict_tokens,
                        normalize=config.normalize, sign=-1,
                    )

                    flipped = v_orig != v_steered
                    flips.append(int(flipped))
                    grad_norms.append(float(np.linalg.norm(grad_b)))

                    record = {
                        "pair_id":         pair.pair_id,
                        "layer_idx":       layer_idx,
                        "verdict_orig":    v_orig,
                        "verdict_steered": v_steered,
                        "flipped":         flipped,
                        "nll_orig":        nll_orig,
                        "grad_norm":       float(np.linalg.norm(grad_b)),
                    }

                    # --- sanity check: h + α · ∇NLL(first_pos_token) ---
                    if sanity_check:
                        first_token = next(
                            t for t in config.verdict_tokens if t != pair.second_token_ab
                        )
                        try:
                            data_alt = compute_layer_outputs(
                                model, tokenizer,
                                pair.prompt_ab,
                                first_token,
                                [layer_idx],
                            )
                            if (layer_idx in data_alt
                                    and data_alt[layer_idx]["gradient"] is not None):
                                grad_a = data_alt[layer_idx]["gradient"]
                                v_steered_alt, _ = get_verdict_with_gradient_steer(
                                    model, tokenizer, pair.prompt_ab,
                                    layer_idx, act, grad_a, config.alpha,
                                    config.verdict_tokens,
                                    normalize=config.normalize, sign=+1,
                                )
                                flipped_alt = v_orig != v_steered_alt
                                flips_alt.append(int(flipped_alt))
                                record["verdict_steered_alt"] = v_steered_alt
                                record["flipped_alt"] = flipped_alt
                        except Exception as e:
                            print(f"  [skip-alt] pair {pair.pair_id} layer {layer_idx}: {e}")

                    f.write(json.dumps(record) + "\n")
                    f.flush()

                except Exception as e:
                    print(f"  [skip] pair {pair.pair_id} layer {layer_idx}: {e}")

            if flips:
                entry = {
                    "lasr":           float(np.mean(flips)),
                    "mean_grad_norm": float(np.mean(grad_norms)),
                    "n_pairs":        len(flips),
                }
                if flips_alt:
                    entry["lasr_alt"] = float(np.mean(flips_alt))
                lasr_per_layer[layer_idx] = entry
                alt_str = (f"  LASR_alt={entry['lasr_alt']:.3f}"
                           if "lasr_alt" in entry else "")
                print(f"  layer={layer_idx:3d}  LASR={entry['lasr']:.3f}{alt_str}  "
                      f"grad_norm={np.mean(grad_norms):.2f}  n={len(flips)}")

            _clear_cache()

    with open(out / "exp1_summary.json", "w") as f:
        json.dump(lasr_per_layer, f, indent=2)

    if not lasr_per_layer:
        print("  [warn] No valid results — cannot determine peak layer.")
        return config.target_layer

    peak_layer = max(lasr_per_layer, key=lambda l: lasr_per_layer[l]["lasr"])
    print(f"\n  Peak layer: {peak_layer}  "
          f"(LASR={lasr_per_layer[peak_layer]['lasr']:.3f})")
    return peak_layer


# ---------------------------------------------------------------------------
# Experiment 2: Swap antisymmetry
# ---------------------------------------------------------------------------

def run_experiment_2(config: GradientProbeConfig, model, tokenizer, swap_pairs,
                     target_layer: int):
    """
    cosine( g_AB , -g_BA ) at target_layer for each pair.
    Reports mean cosine and distribution.
    """
    out = _out(config)
    print(f"\n=== Experiment 2: Swap Antisymmetry (layer={target_layer}) ===")

    cosines = []

    with open(out / "exp2_antisymmetry.jsonl", "w") as f:
        for pair in tqdm(swap_pairs, desc="pairs"):
            try:
                result = compute_swap_antisymmetry(
                    model, tokenizer,
                    pair.prompt_ab, pair.prompt_ba,
                    target_layer,
                    pair.second_token_ab, pair.second_token_ba,
                )
                result["pair_id"] = pair.pair_id
                # Replace NaN with null for valid JSON serialization
                safe = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                        for k, v in result.items()}
                f.write(json.dumps(safe) + "\n")
                f.flush()

                if not np.isnan(result["cosine"]):
                    cosines.append(result["cosine"])

            except Exception as e:
                print(f"  [skip] pair {pair.pair_id}: {e}")

            _clear_cache()

    summary = {
        "target_layer":    target_layer,
        "n_pairs":         len(cosines),
        "cosine_mean":     float(np.mean(cosines)) if cosines else float("nan"),
        "cosine_std":      float(np.std(cosines))  if cosines else float("nan"),
        "cosine_median":   float(np.median(cosines)) if cosines else float("nan"),
        "pct_above_0.5":   float(np.mean(np.array(cosines) > 0.5)) if cosines else float("nan"),
        "pct_above_0.8":   float(np.mean(np.array(cosines) > 0.8)) if cosines else float("nan"),
    }
    with open(out / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  cosine mean={summary['cosine_mean']:.3f}  "
          f"std={summary['cosine_std']:.3f}  "
          f"median={summary['cosine_median']:.3f}")
    print(f"  >0.5: {100*summary['pct_above_0.5']:.1f}%   "
          f">0.8: {100*summary['pct_above_0.8']:.1f}%")


# ---------------------------------------------------------------------------
# Experiment 3: Antisymmetry as positional bias classifier
# ---------------------------------------------------------------------------

def run_experiment_3(config: GradientProbeConfig, model, tokenizer, swap_pairs,
                     target_layer: int):
    """
    1. Label each pair biased/unbiased via the swap verdict test.
    2. Compute antisymmetry score for each pair.
    3. Save scores + labels for downstream ROC analysis.
    """
    out = _out(config)
    print(f"\n=== Experiment 3: Antisymmetry as Bias Classifier (layer={target_layer}) ===")

    # Step 1: classify pairs by swap verdict
    print("  Classifying pairs (running judge on both orderings)...")
    bias_labels = classify_positional_bias(
        model, tokenizer, swap_pairs, config.verdict_tokens
    )
    label_map = {r["pair_id"]: r for r in bias_labels}

    n_biased   = sum(r["biased"] for r in bias_labels)
    n_unbiased = len(bias_labels) - n_biased
    print(f"  biased: {n_biased}  unbiased: {n_unbiased}  "
          f"({100*n_biased/max(len(bias_labels),1):.1f}% biased)")

    # Step 2: compute antisymmetry for each pair
    print("  Computing antisymmetry scores...")
    records = []
    with open(out / "exp3_classifier.jsonl", "w") as f:
        for pair in tqdm(swap_pairs, desc="pairs"):
            try:
                anti = compute_swap_antisymmetry(
                    model, tokenizer,
                    pair.prompt_ab, pair.prompt_ba,
                    target_layer,
                    pair.second_token_ab, pair.second_token_ba,
                )
                label_info = label_map.get(pair.pair_id, {})
                record = {
                    "pair_id":      pair.pair_id,
                    "cosine":       anti["cosine"],
                    "g_ab_norm":    anti.get("g_ab_norm"),
                    "g_ba_norm":    anti.get("g_ba_norm"),
                    "biased":       label_info.get("biased"),
                    "verdict_ab":   label_info.get("verdict_ab"),
                    "verdict_ba":   label_info.get("verdict_ba"),
                    "human_pref":   pair.human_pref,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                records.append(record)

            except Exception as e:
                print(f"  [skip] pair {pair.pair_id}: {e}")

            _clear_cache()

    # Quick summary: mean antisymmetry for biased vs unbiased
    biased_cos   = [r["cosine"] for r in records if r.get("biased") and not np.isnan(r.get("cosine", float("nan")))]
    unbiased_cos = [r["cosine"] for r in records if r.get("biased") is False and not np.isnan(r.get("cosine", float("nan")))]

    summary = {
        "target_layer":           target_layer,
        "n_biased":               n_biased,
        "n_unbiased":             n_unbiased,
        "cosine_biased_mean":     float(np.mean(biased_cos))   if biased_cos   else float("nan"),
        "cosine_biased_std":      float(np.std(biased_cos))    if biased_cos   else float("nan"),
        "cosine_unbiased_mean":   float(np.mean(unbiased_cos)) if unbiased_cos else float("nan"),
        "cosine_unbiased_std":    float(np.std(unbiased_cos))  if unbiased_cos else float("nan"),
    }
    with open(out / "exp3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  cosine (biased):   {summary['cosine_biased_mean']:.3f} ± {summary['cosine_biased_std']:.3f}")
    print(f"  cosine (unbiased): {summary['cosine_unbiased_mean']:.3f} ± {summary['cosine_unbiased_std']:.3f}")
    print(f"  (ROC analysis: run scripts/plot_exp3_roc.py on exp3_classifier.jsonl)")


# ---------------------------------------------------------------------------
# Experiment 4: Dose-response at target layer
# ---------------------------------------------------------------------------

def run_experiment_4(config: GradientProbeConfig, model, tokenizer, swap_pairs,
                     target_layer: int):
    """
    Vary alpha across config.alpha_scales at target_layer.
    For each alpha:
      - gradient direction: steer by alpha * g/||g||
      - random baseline:    steer by alpha * random_orth/||random_orth||
    Record flip rate and ΔNLL(second_position_token) for both.
    """
    out = _out(config)
    rng = np.random.default_rng(config.seed)
    print(f"\n=== Experiment 4: Dose-Response (layer={target_layer}) ===")
    print(f"  alpha scales: {config.alpha_scales}")

    # Pre-compute activations and gradients at target_layer for all pairs
    print("  Pre-computing gradients at target layer...")
    pair_data = {}
    for pair in tqdm(swap_pairs, desc="extract"):
        try:
            data = compute_layer_outputs(
                model, tokenizer,
                pair.prompt_ab, pair.second_token_ab,
                [target_layer],
            )
            if target_layer in data and data[target_layer]["gradient"] is not None:
                pair_data[pair.pair_id] = {
                    "pair":       pair,
                    "activation": data[target_layer]["activation"],
                    "gradient":   data[target_layer]["gradient"],
                    "nll_orig":   data[target_layer]["nll"],
                }
        except Exception as e:
            print(f"  [skip] pair {pair.pair_id}: {e}")
        _clear_cache()

    print(f"  Valid pairs with gradients: {len(pair_data)}")

    alpha_summaries = []

    with open(out / "exp4_dose_response.jsonl", "w") as f:
        for alpha in tqdm(config.alpha_scales, desc="alpha"):
            grad_flips, rand_flips = [], []

            for pid, pd in pair_data.items():
                pair = pd["pair"]
                act  = pd["activation"]
                grad = pd["gradient"]

                try:
                    # Gradient direction
                    v_orig = get_verdict(model, tokenizer, pair.prompt_ab,
                                        config.verdict_tokens)
                    v_grad, _ = get_verdict_with_gradient_steer(
                        model, tokenizer, pair.prompt_ab,
                        target_layer, act, grad, alpha, config.verdict_tokens,
                        normalize=config.normalize, sign=-1,
                    )
                    grad_flip = int(v_orig != v_grad)
                    grad_flips.append(grad_flip)

                    # Random orthogonal baseline
                    rand_result = random_direction_baseline(
                        model, tokenizer, pair.prompt_ab,
                        target_layer, act, grad, alpha,
                        v_orig,
                        n_trials=config.n_random_baselines,
                        rng=rng,
                        verdict_tokens=config.verdict_tokens,
                    )
                    rand_flips.append(rand_result["flip_rate_mean"])

                    record = {
                        "pair_id":          pid,
                        "alpha":            alpha,
                        "verdict_orig":     v_orig,
                        "verdict_grad":     v_grad,
                        "grad_flip":        grad_flip,
                        "rand_flip_mean":   rand_result["flip_rate_mean"],
                        "rand_flip_std":    rand_result["flip_rate_std"],
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                except Exception as e:
                    print(f"  [skip] pair {pid} alpha {alpha}: {e}")

                _clear_cache()

            if grad_flips:
                alpha_summaries.append({
                    "alpha":               alpha,
                    "grad_flip_rate":      float(np.mean(grad_flips)),
                    "rand_flip_rate_mean": float(np.mean(rand_flips)),
                    "rand_flip_rate_std":  float(np.std(rand_flips)),
                    "specificity":         float(np.mean(grad_flips) - np.mean(rand_flips)),
                    "n_pairs":             len(grad_flips),
                })
                print(f"  alpha={alpha:.2f}  "
                      f"grad_flip={np.mean(grad_flips):.3f}  "
                      f"rand_flip={np.mean(rand_flips):.3f}  "
                      f"specificity={np.mean(grad_flips)-np.mean(rand_flips):+.3f}")

    with open(out / "exp4_summary.json", "w") as f:
        json.dump(alpha_summaries, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="meta-llama/Llama-3.1-70B-Instruct")
    p.add_argument("--device",        default="cuda")
    quant = p.add_mutually_exclusive_group()
    quant.add_argument("--4bit",   dest="quant", action="store_const", const="4bit", default="4bit")
    quant.add_argument("--8bit",   dest="quant", action="store_const", const="8bit")
    quant.add_argument("--no-4bit", dest="quant", action="store_const", const="none")
    p.add_argument("--dtype",         default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--n-pairs",       type=int, default=200)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--alpha",         type=float, default=1.0)
    p.add_argument("--alpha-scales",  nargs="+", type=float,
                   default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    p.add_argument("--layer-step",    type=int, default=4,
                   help="Probe every Nth layer in Exp 1")
    p.add_argument("--target-layer",  type=int, default=-1,
                   help="Peak layer for Exp 2/3/4; auto-set from Exp 1 if running all")
    p.add_argument("--n-random",      type=int, default=5,
                   help="Random baseline trials per pair in Exp 4")
    p.add_argument("--output-dir",    default="results/gradient_probe")
    p.add_argument("--experiment",    default="all",
                   choices=["1", "2", "3", "4", "all"])
    p.add_argument("--filter-biased", action="store_true",
                   help="Pre-screen pairs and keep only positionally biased ones")
    p.add_argument("--pool-multiplier", type=int, default=4,
                   help="Pool size = n_pairs * pool_multiplier for bias screening")
    p.add_argument("--bias-cache",    default=None,
                   help="Path to cache bias labels (JSONL) to avoid re-running verdicts")
    p.add_argument("--bias-ratio",    type=float, default=0.5,
                   help="Fraction of selected pairs that are biased (default 0.5 for balanced ROC)")
    p.add_argument("--normalize",     action="store_true",
                   help="Apply LatentSafety distribution normalization to perturbation "
                        "(rescales perturbed activation to match hidden state mean/std)")
    p.add_argument("--exp1-sanity",   action="store_true",
                   help="Exp 1 sanity check: also run h + α·∇NLL(first_pos_token) "
                        "alongside the default h - α·∇NLL(second_pos_token) and compare LASR")
    args = p.parse_args()

    config = GradientProbeConfig(
        model_name=args.model,
        device=args.device,
        load_in_4bit=(args.quant == "4bit"),
        load_in_8bit=(args.quant == "8bit"),
        dtype=args.dtype,
        layer_sweep_step=args.layer_step,
        target_layer=args.target_layer,
        normalize=args.normalize,
        alpha=args.alpha,
        alpha_scales=args.alpha_scales,
        n_pairs=args.n_pairs,
        n_random_baselines=args.n_random,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"Loading {config.model_name}...")
    model, tokenizer = load_model(config)

    # Exp 1/2/4: all-biased pairs (maximise signal for layer sweep, antisymmetry, dose-response)
    # Exp 3:     50/50 biased+unbiased pairs (need both groups for the classifier ROC)
    _common = dict(
        tokenizer=tokenizer,
        n=config.n_pairs,
        seed=config.seed,
        template_id=config.template_id,
        criterion=config.criterion,
        pool_multiplier=args.pool_multiplier,
    )

    print(f"Loading {config.n_pairs} biased swap pairs (Exp 1/2/4)...")
    swap_pairs_biased = load_swap_pairs(
        **_common,
        filter_biased=args.filter_biased,
        model=model if args.filter_biased else None,
        bias_cache_path=args.bias_cache,
        bias_ratio=1.0,   # all biased
    )

    # Only load the mixed pool if Exp 3 is being run
    swap_pairs_mixed = None
    if args.experiment in ("3", "all"):
        print(f"Loading {config.n_pairs} mixed swap pairs (Exp 3, 50/50)...")
        swap_pairs_mixed = load_swap_pairs(
            **_common,
            filter_biased=args.filter_biased,
            model=model if args.filter_biased else None,
            bias_cache_path=args.bias_cache,
            bias_ratio=args.bias_ratio,   # default 0.5
        )

    target_layer = config.target_layer

    if args.experiment in ("1", "all"):
        target_layer = run_experiment_1(
            config, model, tokenizer, swap_pairs_biased,
            sanity_check=args.exp1_sanity,
        )
        (_out(config) / "peak_layer.txt").write_text(str(target_layer))

    # Allow manual override for experiments 2-4 when running without exp 1
    if args.experiment != "all" and config.target_layer != -1:
        target_layer = config.target_layer

    if args.experiment in ("2", "all"):
        run_experiment_2(config, model, tokenizer, swap_pairs_biased, target_layer)

    if args.experiment in ("3", "all"):
        run_experiment_3(config, model, tokenizer, swap_pairs_mixed, target_layer)

    if args.experiment in ("4", "all"):
        run_experiment_4(config, model, tokenizer, swap_pairs_biased, target_layer)
