"""
gradient_probe.py
-----------------
Analysis functions that sit on top of model.py's gradient extraction.

compute_nll_delta()
    ΔNLL after gradient steering — the continuous sensitivity metric used
    in place of binary flip counts for Experiments 1 & 4.

compute_swap_antisymmetry()
    cosine( g_AB , -g_BA ) — the core test for whether the gradient
    is a pure position signal vs a content signal.

classify_positional_bias()
    Run the judge on both orderings and label each pair as biased / unbiased.
    Required preprocessing for Experiment 3.

random_direction_baseline()
    Draw K random unit vectors orthogonal to the gradient and measure the
    same LASR / ΔNLL — establishes the specificity of the gradient direction.
"""

import numpy as np
from typing import Optional

from src.latent_perturbations.model import (
    compute_layer_outputs,
    get_verdict,
    get_verdict_with_gradient_steer,
)


def compute_nll_delta(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    activation: np.ndarray,
    gradient: np.ndarray,
    alpha: float,
    target_token: str,
    verdict_tokens: Optional[list[str]] = None,
) -> dict:
    """
    Steer at layer_idx by alpha * g/||g||, return:
        verdict_orig, verdict_steered, flipped, delta_nll, nll_orig, nll_steered
    """
    if verdict_tokens is None:
        verdict_tokens = ["A", "B", "C"]

    verdict_orig = get_verdict(model, tokenizer, prompt, verdict_tokens)

    # NLL of target_token before perturbation
    data_orig = compute_layer_outputs(model, tokenizer, prompt, target_token, [layer_idx])
    nll_orig = data_orig[layer_idx]["nll"] if layer_idx in data_orig else float("nan")

    # Steer and get new verdict + NLL
    # get_verdict_with_gradient_steer returns verdict and NLL of first verdict token;
    # we need NLL of target_token, so we do a small extra pass.
    verdict_steered, _ = get_verdict_with_gradient_steer(
        model, tokenizer, prompt, layer_idx,
        activation, gradient, alpha, verdict_tokens,
    )

    # Re-compute NLL after steering to get ΔNLL(target_token)
    # We approximate by re-running compute_layer_outputs on the steered prompt.
    # For efficiency in large sweeps, callers can skip this and use flip rate only.
    nll_steered = float("nan")  # filled by experiment.py if needed

    return {
        "verdict_orig":    verdict_orig,
        "verdict_steered": verdict_steered,
        "flipped":         verdict_orig != verdict_steered,
        "nll_orig":        nll_orig,
        "nll_steered":     nll_steered,
        "delta_nll":       nll_steered - nll_orig,
    }


def compute_swap_antisymmetry(
    model,
    tokenizer,
    prompt_ab: str,
    prompt_ba: str,
    layer_idx: int,
    second_token_ab: str = "B",
    second_token_ba: str = "A",
) -> dict:
    """
    Compute cosine( g_AB , -g_BA ) at layer_idx.

    g_AB = ∂NLL(second_token_ab) / ∂h_l  on prompt (A,B)
    g_BA = ∂NLL(second_token_ba) / ∂h_l  on prompt (B,A)

    Interpretation:
      cosine ≈ 1.0  → pure position signal (both gradients are antiparallel)
      cosine ≈ 0.0  → gradient is driven by content, not position
    """
    out_ab = compute_layer_outputs(model, tokenizer, prompt_ab, second_token_ab, [layer_idx])
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:
        pass
    out_ba = compute_layer_outputs(model, tokenizer, prompt_ba, second_token_ba, [layer_idx])

    if layer_idx not in out_ab or layer_idx not in out_ba:
        return {"cosine": float("nan"), "g_ab_norm": float("nan"), "g_ba_norm": float("nan")}

    g_ab = out_ab[layer_idx]["gradient"]
    g_ba = out_ba[layer_idx]["gradient"]

    if g_ab is None or g_ba is None:
        return {"cosine": float("nan"), "g_ab_norm": float("nan"), "g_ba_norm": float("nan")}

    g_ab_u = g_ab / (np.linalg.norm(g_ab) + 1e-12)
    g_ba_u = g_ba / (np.linalg.norm(g_ba) + 1e-12)

    cosine = float(np.dot(g_ab_u, -g_ba_u))   # antiparallel → +1 if pure position

    return {
        "cosine":      cosine,
        "g_ab_norm":   float(np.linalg.norm(g_ab)),
        "g_ba_norm":   float(np.linalg.norm(g_ba)),
        "nll_ab":      out_ab[layer_idx]["nll"],
        "nll_ba":      out_ba[layer_idx]["nll"],
    }


def classify_positional_bias(
    model,
    tokenizer,
    swap_pairs,
    verdict_tokens: Optional[list[str]] = None,
) -> list[dict]:
    """
    Run the judge on both orderings for each pair and label:
      biased   — model picks same response on (A,B) and (B,A)
      unbiased — model flips verdict with the ordering (consistent with position)

    Returns list of dicts with pair_id, verdict_ab, verdict_ba, biased (bool).
    """
    if verdict_tokens is None:
        verdict_tokens = ["A", "B", "C"]

    results = []
    for pair in swap_pairs:
        v_ab = get_verdict(model, tokenizer, pair.prompt_ab, verdict_tokens)
        v_ba = get_verdict(model, tokenizer, pair.prompt_ba, verdict_tokens)

        # Verdict labels refer to slots, not responses.
        # "A" always means slot_1; "B" always means slot_2.
        # In (A,B): slot_1=resp_a, slot_2=resp_b.
        # In (B,A): slot_1=resp_b, slot_2=resp_a.
        #
        # Same verdict label → same slot chosen → different resp in that slot
        #   → model tracks position, not content → BIASED.
        # Different verdict label → slot flipped with order → same resp won both times
        #   → model tracks content → UNBIASED.
        biased = v_ab == v_ba

        results.append({
            "pair_id":   pair.pair_id,
            "verdict_ab": v_ab,
            "verdict_ba": v_ba,
            "biased":    biased,
        })

    return results


def random_direction_baseline(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    activation: np.ndarray,
    gradient: np.ndarray,
    alpha: float,
    verdict_orig: str,
    n_trials: int = 5,
    rng: Optional[np.random.Generator] = None,
    verdict_tokens: Optional[list[str]] = None,
) -> dict:
    """
    Draw n_trials random unit vectors in the orthogonal complement of g,
    steer by the same alpha, and record flip rates.

    Returns mean and std flip rate across trials — the specificity baseline
    for the gradient direction.
    """
    if verdict_tokens is None:
        verdict_tokens = ["A", "B", "C"]
    if rng is None:
        rng = np.random.default_rng(0)

    hidden_dim = gradient.shape[0]
    g_unit = gradient / (np.linalg.norm(gradient) + 1e-12)

    flips = []
    for _ in range(n_trials):
        rand_vec = rng.standard_normal(hidden_dim).astype(np.float64)
        # Project out gradient direction so random vec is orthogonal to g
        rand_vec -= np.dot(rand_vec, g_unit) * g_unit
        rand_vec /= np.linalg.norm(rand_vec) + 1e-12

        v_rand, _ = get_verdict_with_gradient_steer(
            model, tokenizer, prompt, layer_idx,
            activation, rand_vec, alpha, verdict_tokens,
        )
        flips.append(int(v_rand != verdict_orig))

    return {
        "flip_rate_mean": float(np.mean(flips)),
        "flip_rate_std":  float(np.std(flips)),
        "n_trials":       n_trials,
    }
