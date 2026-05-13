"""
model.py
--------
Model loading and gradient-enabled activation extraction.

Core function: compute_layer_outputs()
  Uses the activation-as-leaf approach (same as LatentSafety gasa_attack):

  Pass 1 — no-grad forward: capture h_l at every requested layer.
  Pass 2 — per layer, inject h_l as a float32 leaf tensor at that layer via
            a forward hook, run the rest of the model with grad enabled,
            compute NLL(target_token), backprop → h_leaf.grad is the gradient.

  This avoids backpropping through bitsandbytes 4bit kernels (which cause
  cudaErrorLaunchFailure). Only layers above the injection point (float16/bf16
  activations) are in the backward graph.

Perturbation: get_verdict_with_gradient_steer()
  Replaces h_l with h_l + sign * alpha * g/||g|| via forward hook.
    sign=-1 (default): h_l - alpha * g/||g||, g = ∇NLL(second_pos_token)
      Subtracting the gradient moves toward second_pos_token being more likely.
    sign=+1 (paper convention): h_l + alpha * g/||g||, g = ∇NLL(current_output)
      Adding ∇NLL(current_output) makes the current output less likely.
  Both steer away from the current verdict. Use sign=-1/target=second_pos_token
  or sign=+1/target=current_output — they are equivalent in expectation.

  normalize=True applies the LatentSafety distribution normalization:
    δ' = μ(h) + (δ - μ(δ)) / (σ(δ) + ε) * σ(h)
  where δ = h + sign * alpha * g/||g|| (the full perturbed vector).
  This rescales δ to match h's mean and std, keeping activations in-distribution.
"""

import numpy as np
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.latent_perturbations.config import GradientProbeConfig


def load_model(config: GradientProbeConfig):
    """Load model and tokenizer with optional 4-bit quantization."""
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    kwargs = {"torch_dtype": dtype}

    if config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    elif config.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = config.device

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_num_layers(model) -> int:
    return len(model.model.layers)


def resolve_layer_indices(model, config: GradientProbeConfig) -> list[int]:
    if config.layer_indices:
        return config.layer_indices
    n = get_num_layers(model)
    return list(range(0, n, config.layer_sweep_step))


def _clear_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def compute_layer_outputs(
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    layer_indices: list[int],
) -> dict[int, dict]:
    """
    For each layer in layer_indices, returns:
        {
          'activation': np.ndarray [hidden_dim]
          'gradient':   np.ndarray [hidden_dim]  — ∂NLL(target_token)/∂h_l
          'nll':        float
        }

    Two-pass approach (activation-as-leaf):
      Pass 1: no-grad forward — capture activations at all requested layers.
      Pass 2: for each layer, re-run the model with a float32 leaf tensor
              injected at that layer so backward only flows through the
              float16 layers above it, avoiding 4bit kernel backward issues.
    """
    device = next(model.parameters()).device
    target_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    last_pos = enc["input_ids"].shape[1] - 1

    # --- Pass 1: extract activations (no grad) ---
    activations: dict[int, np.ndarray] = {}
    handles = []
    for l in layer_indices:
        def _make_capture(li: int):
            def _hook(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                activations[li] = h.detach()[0, last_pos, :].float().cpu().numpy()
            return _hook
        handles.append(model.model.layers[l].register_forward_hook(_make_capture(l)))

    with torch.no_grad():
        model(**enc)

    for h in handles:
        h.remove()

    # --- Pass 2: gradient per layer via activation-as-leaf ---
    result = {}
    missing = []

    for l in layer_indices:
        if l not in activations:
            missing.append(l)
            continue

        act_np = activations[l]
        # Leaf tensor in float32 — gradient will accumulate here.
        # Injected at layer l so backward only flows through layers l+1..end.
        h_leaf = torch.tensor(
            act_np, dtype=torch.float32, device=device, requires_grad=True
        )

        def _inject(module, inp, output, _leaf=h_leaf, _lp=last_pos):
            h = output[0] if isinstance(output, tuple) else output
            h_new = h.clone()
            lp = min(_lp, h_new.shape[1] - 1)
            h_new[0, lp, :] = _leaf.to(h_new.dtype)
            return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

        handle = model.model.layers[l].register_forward_hook(_inject)
        nll_val = float("nan")
        grad = None
        try:
            with torch.enable_grad():
                out = model(**enc)
                logits = out.logits[0, last_pos, :].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                nll_val = float(-log_probs[target_id].item())
                (-log_probs[target_id]).backward()
            if h_leaf.grad is not None:
                grad = h_leaf.grad.float().cpu().numpy()
            else:
                missing.append(l)
        except Exception as e:
            print(f"  [warn] gradient failed at layer {l}: {e}")
            missing.append(l)
        finally:
            handle.remove()
            del out, h_leaf
            _clear_cuda()

        result[l] = {"activation": act_np, "gradient": grad, "nll": nll_val}

    if missing:
        print(f"  [warn] no gradient at layers {missing}")

    return result


def get_verdict(
    model,
    tokenizer,
    prompt: str,
    verdict_tokens: Optional[list[str]] = None,
) -> str:
    """Standard next-token verdict (no perturbation)."""
    if verdict_tokens is None:
        verdict_tokens = ["A", "B", "C"]
    device = next(model.parameters()).device
    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in verdict_tokens]
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]
    restricted = {t: logits[tid].item() for t, tid in zip(verdict_tokens, token_ids)}
    return max(restricted, key=restricted.get)


def get_verdict_with_gradient_steer(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    activation: np.ndarray,
    gradient: np.ndarray,
    alpha: float,
    verdict_tokens: Optional[list[str]] = None,
    normalize: bool = False,
    sign: int = -1,
    norm_type: str = "unit",
) -> tuple[str, float]:
    """
    Steer activation at layer_idx, return (verdict, new_nll).

    sign=-1 (default): h' = h - alpha * g_norm, g = ∇NLL(second_pos_token)
    sign=+1:           h' = h + alpha * g_norm, g = ∇NLL(current_output)
    norm_type="unit":  g_norm = g/||g||  (unit vector, total magnitude = 1)
    norm_type="sign":  g_norm = sign(g)  (FGSM-style, total magnitude = sqrt(hidden_dim))
    normalize=True: rescale h' to match h's mean/std (LatentSafety normalization).
    """
    if verdict_tokens is None:
        verdict_tokens = ["A", "B", "C"]

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in verdict_tokens]

    if norm_type == "sign":
        g_normalized = np.sign(gradient)
    else:
        g_normalized = gradient / (np.linalg.norm(gradient) + 1e-12)

    delta_raw = activation + sign * alpha * g_normalized  # full perturbed vector

    if normalize:
        h_mean = float(activation.mean())
        h_std  = float(activation.std())
        d_mean = float(delta_raw.mean())
        d_std  = float(delta_raw.std()) + 1e-12
        patched_np = h_mean + (delta_raw - d_mean) / d_std * h_std
    else:
        patched_np = delta_raw

    patched = torch.tensor(patched_np.astype(np.float32), device=device).to(dtype)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    last_pos = enc["input_ids"].shape[1] - 1

    def _hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        lp = min(last_pos, h.shape[1] - 1)
        h[0, lp, :] = patched.to(h.dtype)
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = model.model.layers[layer_idx].register_forward_hook(_hook)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]
    handle.remove()

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    restricted = {t: logits[token_ids[i]].item() for i, t in enumerate(verdict_tokens)}
    verdict = max(restricted, key=restricted.get)
    new_nll = float(-log_probs[token_ids[0]].item())

    return verdict, new_nll
