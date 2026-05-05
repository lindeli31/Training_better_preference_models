"""
Perturbation sensitivity test.

Two modes with different noise injection points:

DECODE MODE — last-hidden-layer hook
    Registers a forward hook on the last transformer block that adds Gaussian
    noise to every token's hidden state at each generation step. Tests whether
    the model still produces coherent text under whole-sequence representation
    noise.

JUDGE MODE — input token embedding perturbation
    The full judge prompt (system text + question + answer A + answer B) is
    tokenized once. The token span belonging to one answer is located using the
    tokenizer's offset mapping. Gaussian noise is then added exclusively to the
    embedding vectors of those tokens before the forward pass runs at all. The
    model receives `inputs_embeds` (the modified embeddings) instead of
    `input_ids`, so the noise originates at the earliest possible point — before
    any attention computation. Everything outside the target answer's span is
    left untouched.

    Limitation: self-attention propagates the noise from the perturbed positions
    into all other positions as it passes through each layer, so the separation
    is not perfectly clean by the final layer. But the origin of the perturbation
    is surgically confined to the target answer's tokens.

Usage:
    python perturb_hidden.py --mode decode   # test noisy decoding
    python perturb_hidden.py --mode judge    # test judge preference stability
    python perturb_hidden.py --mode both     # run both
"""

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Small models that comfortably fit in 16 GB on Apple Silicon.
# Llama-3.2-1B-Instruct (~1B params, ~2 GB in float16) is the default.
# Requires a HuggingFace account with Llama access; run `huggingface-cli login` first.
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

NOISE_SCALES = [0.0, 0.01, 0.05, 0.1, 0.3, 1.0]


# ── noise hook ────────────────────────────────────────────────────────────────
class HiddenLayerNoise:
    """Register as a forward hook on a transformer layer to add Gaussian noise."""

    def __init__(self, scale: float = 0.1):
        self.scale = scale
        self.handle = None

    def hook_fn(self, module, input, output):
        if self.scale == 0.0:
            return output
        # output is usually a tuple; first element is the hidden states tensor
        hidden = output[0] if isinstance(output, tuple) else output
        noise = torch.randn_like(hidden) * self.scale
        noisy = hidden + noise
        if isinstance(output, tuple):
            return (noisy,) + output[1:]
        return noisy

    def register(self, layer):
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(model_name: str):
    print(f"Loading {model_name} on {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # float16 halves memory vs float32; safe for inference on MPS and CPU.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(DEVICE)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_last_layer(model):
    """Return the last transformer block regardless of model family."""
    # Llama / Mistral / Gemma style (LlamaForCausalLM, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[-1]
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[-1]
    # GPT-Neo / GPT-J style
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return model.transformer.blocks[-1]
    raise ValueError("Could not identify the last hidden layer automatically.")


def perturb_generate(model, tokenizer, prompt: str, scale: float,
                     max_new_tokens: int = 60) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    noiser = HiddenLayerNoise(scale=scale)
    last_layer = get_last_layer(model)
    noiser.register(last_layer)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy — deterministic baseline
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        noiser.remove()
    # strip the prompt tokens from the output
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of *text* under the model (no noise)."""
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss))


# ── mode: decode ─────────────────────────────────────────────────────────────

def run_decode_test(model, tokenizer, n_prompts: int = 3):
    """
    For several noise levels, generate text from a fixed prompt and report
    perplexity of the generated continuation.
    """
    test_prompts = [
        "The capital of France is",
        "Explain the concept of entropy in thermodynamics:",
        "Once upon a time in a distant land,",
    ][:n_prompts]

    print("\n" + "=" * 70)
    print("MODE: DECODE — noisy last-hidden-layer generation")
    print("=" * 70)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt!r}")
        print("-" * 60)
        for scale in NOISE_SCALES:
            text = perturb_generate(model, tokenizer, prompt, scale=scale)
            ppl = perplexity(model, tokenizer, text) if text.strip() else float("inf")
            print(f"  noise={scale:.2f}  ppl={ppl:7.1f}  text={text[:80]!r}")


# ── mode: judge ───────────────────────────────────────────────────────────────

def find_answer_token_span(prompt: str, answer: str, marker: str,
                           tokenizer) -> tuple[int, int]:
    """
    Return (tok_start, tok_end) — the half-open token index range of `answer`
    within the tokenized `prompt`.

    Strategy: ask the tokenizer for its offset mapping, which gives the
    (char_start, char_end) character span for every token in the sequence.
    We locate the answer's character span in the prompt string, then read off
    the corresponding token indices. This avoids context-dependent tokenization
    issues that arise when trying to tokenize sub-strings independently.
    """
    char_start = prompt.index(marker) + len(marker)
    char_end = char_start + len(answer)

    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token

    # first token whose character span overlaps the answer text
    tok_start = next(
        (i for i, (s, e) in enumerate(offsets) if e > char_start),
        len(offsets),
    )
    # first token that begins at or after the end of the answer
    tok_end = next(
        (i for i, (s, e) in enumerate(offsets) if s >= char_end),
        len(offsets),
    )
    return tok_start, tok_end


_JUDGE_SYSTEM = (
    'You are an impartial judge. Given a question and two answers, '
    'choose which answer is better. Reply with ONLY "A" or "B".'
)


def make_judge_prompt(tokenizer, question: str, answer_a: str, answer_b: str) -> str:
    """
    Build the judge prompt using the model's chat template so that instruct
    models receive the exact format they were fine-tuned on.

    The user message embeds the markers "Answer A: " and "Answer B: " which
    find_answer_token_span later uses to locate the answer token spans inside
    the formatted string.
    """
    user_content = (
        f"Question: {question}\n\n"
        f"Answer A: {answer_a}\n\n"
        f"Answer B: {answer_b}\n\n"
        'Which answer is better? Reply with ONLY "A" or "B".'
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def parse_preference(raw: str) -> str:
    """
    Extract "A" or "B" from the model's raw text output.

    Priority order:
    1. Explicit "Answer A" / "Answer B" phrases  — handles "Answer B" which
       starts with the letter A and would be misread by a first-char check.
       When both appear, the last mention wins.
    2. Text begins with a bare A or B (e.g. "A", "A.", "B is better").
    3. Verb + letter phrases: "choose A", "prefer B", etc.
    4. Fallback: "?" (ambiguous / no clear preference).
    """
    text = raw.strip()
    if not text:
        return "?"
    lower = text.lower()

    # 1. "Answer A" / "Answer B"
    positions_a = [m.start() for m in re.finditer(r'\banswer\s+a\b', lower)]
    positions_b = [m.start() for m in re.finditer(r'\banswer\s+b\b', lower)]
    if positions_a or positions_b:
        if positions_a and not positions_b:
            return "A"
        if positions_b and not positions_a:
            return "B"
        # Both present — last mention wins
        return "A" if max(positions_a) > max(positions_b) else "B"

    # 2. Bare leading letter
    m = re.match(r'^([AaBb])\b', text)
    if m:
        return m.group(1).upper()

    # 3. Verb + letter
    m = re.search(r'\b(?:choose|prefer|select|pick)\s+([AaBb])\b', lower)
    if m:
        return m.group(1).upper()

    return "?"


def judge_preference(model, tokenizer, question: str, answer_a: str,
                     answer_b: str, noise_scale: float,
                     perturb_target: str = "a") -> str:
    """
    Ask the model to judge between A and B with optional input-embedding noise.

    perturb_target:
      "a"    — locate answer A's tokens in the prompt, add Gaussian noise to
               their embedding vectors before any forward pass runs.
      "b"    — same but for answer B's tokens.
      "none" — no noise; runs a standard forward pass via input_ids.

    When noise is applied, the model receives `inputs_embeds` (the embedding
    matrix lookup output, with noise added at the target positions) instead of
    `input_ids`. Because generate() has no input_ids to echo back, its output
    contains only the newly generated tokens.

    Returns "A", "B", or "?" if the model produces neither.
    """
    prompt = make_judge_prompt(tokenizer, question, answer_a, answer_b)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    if perturb_target != "none" and noise_scale > 0.0:
        marker = "Answer A: " if perturb_target == "a" else "Answer B: "
        answer = answer_a if perturb_target == "a" else answer_b

        tok_start, tok_end = find_answer_token_span(prompt, answer, marker, tokenizer)

        # look up token embeddings for the whole prompt — [1, seq_len, hidden_dim]
        embed_layer = model.get_input_embeddings()
        with torch.no_grad():
            embeds = embed_layer(input_ids).clone()

        # add noise only at the target answer's token positions
        embeds[:, tok_start:tok_end, :] += (
            torch.randn_like(embeds[:, tok_start:tok_end, :]) * noise_scale
        )

        with torch.no_grad():
            out = model.generate(
                inputs_embeds=embeds,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # generate() with inputs_embeds returns only the new tokens (no prompt ids to echo)
        gen_ids = out[0]
    else:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # generate() with input_ids prepends the prompt; strip it
        gen_ids = out[0][input_ids.shape[1]:]

    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return parse_preference(raw), raw


def _map_swap(pref: str) -> str:
    """Flip A↔B to translate a preference from swapped-order space back to original."""
    return {"A": "B", "B": "A", "?": "?"}.get(pref, "?")


def run_judge_test(model, tokenizer, n_samples: int = 10):
    """
    Position-debiased judge evaluation with noise sweep.

    For every pair we run the judge twice — once in the original A/B order and
    once with A and B swapped. A preference is only trusted if both orderings
    agree on the same underlying response (consistency check). Pairs where the
    model flips with the swap are position-biased and excluded from the noise
    sweep and accuracy metrics.

    gold_label semantics (helpsteer2):
      "A" → response_a is objectively better (positive score_gap)
      "C" → exact tie (score_gap == 0); excluded from accuracy calculation
    """
    csv_path = DATA_DIR / "helpsteer2_validation.csv"
    if not csv_path.exists():
        print(f"Data file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path).dropna(subset=["prompt", "response_a", "response_b"])
    df = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODE: JUDGE — position-debiased preference stability under noise")
    print("=" * 70)
    print(f"Evaluating {len(df)} Q/A pairs …")
    print("Each pair is judged in both A/B orderings; only consistent pairs")
    print("enter the noise sweep and accuracy metrics.\n")

    flip_counts = {s: 0 for s in NOISE_SCALES[1:]}
    invalid_counts = {s: 0 for s in NOISE_SCALES[1:]}
    n_consistent = 0       # pairs with consistent baseline across both orderings
    n_pos_a_fwd = 0        # how often the model picks position A (forward order)
    n_pos_a_rev = 0        # same for reversed order
    correct_count = 0      # judge matches gold (on non-tie, consistent pairs)
    gold_a_count = 0       # non-tie pairs available for accuracy

    rng = random.Random(42)

    for i, (_, row) in enumerate(df.iterrows()):
        q     = row["prompt"]
        ra    = str(row["response_a"])
        rb    = str(row["response_b"])
        gold  = row.get("gold_label", "?")

        # Randomly assign which response goes in position A for this pair.
        # This makes the position assignment uncorrelated with response quality
        # across the batch, so aggregate position-bias estimates are unconfounded.
        swap = rng.random() < 0.5
        pos_a, pos_b = (rb, ra) if swap else (ra, rb)

        print(f"\n{'─' * 70}")
        print(f"Pair {i+1}  gold={gold}  order={'swapped' if swap else 'original'}")
        print(f"  Q: {q[:120]!r}")

        # ── baseline: forward order ───────────────────────────────────────────
        fwd_pref, fwd_raw = judge_preference(
            model, tokenizer, q, pos_a, pos_b, noise_scale=0.0, perturb_target="none",
        )
        if fwd_pref == "A":
            n_pos_a_fwd += 1

        # ── baseline: reversed order ──────────────────────────────────────────
        rev_pref, rev_raw = judge_preference(
            model, tokenizer, q, pos_b, pos_a, noise_scale=0.0, perturb_target="none",
        )
        if rev_pref == "A":
            n_pos_a_rev += 1
        rev_mapped = _map_swap(rev_pref)   # translate back to forward-order space

        # ── consistency check ─────────────────────────────────────────────────
        consistent = fwd_pref in ("A", "B") and fwd_pref == rev_mapped
        pos_bias_flag = ""
        if fwd_pref in ("A", "B") and rev_mapped in ("A", "B") and fwd_pref != rev_mapped:
            pos_bias_flag = "  ← POSITION BIAS"

        print(f"  fwd:  pref={fwd_pref}  raw={fwd_raw!r}")
        print(f"  rev:  pref={rev_pref} (→{rev_mapped})  raw={rev_raw!r}  "
              f"consistent={consistent}{pos_bias_flag}")

        if not consistent:
            print("  (skipping noise sweep — preference not consistent across orderings)")
            continue

        # ── map position preference → response preference ─────────────────────
        # fwd_pref is in (pos_a, pos_b) space; map back to (response_a, response_b)
        if fwd_pref == "A":
            resp_pref = "B" if swap else "A"
        else:
            resp_pref = "A" if swap else "B"

        # ── accuracy vs gold label ────────────────────────────────────────────
        if gold == "C":
            acc_str = "tie (excluded)"
        else:
            gold_a_count += 1
            correct = resp_pref == gold   # gold is always "A" (response_a wins)
            correct_count += int(correct)
            acc_str = f"correct={correct}  (judge→resp_{resp_pref.lower()}  gold→resp_{gold.lower()})"

        print(f"  debiased pref: resp_{resp_pref.lower()}  {acc_str}")
        n_consistent += 1

        # ── noise sweep (run in forward/assigned order) ───────────────────────
        # perturb_target is in position space: fwd_pref.lower() is "a" or "b"
        target = fwd_pref.lower()
        for scale in NOISE_SCALES[1:]:
            noisy_pref, noisy_raw = judge_preference(
                model, tokenizer, q, pos_a, pos_b,
                noise_scale=scale, perturb_target=target,
            )
            flipped = noisy_pref != fwd_pref and noisy_pref in ("A", "B")
            flag = "  ← FLIP" if flipped else ""
            if noisy_pref == "?":
                invalid_counts[scale] += 1
            elif flipped:
                flip_counts[scale] += 1
            print(f"  noise={scale:.3f}  pref={noisy_pref}  raw={noisy_raw!r}{flag}")

    # ── summary ──────────────────────────────────────────────────────────────
    n = len(df)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"  pairs evaluated  : {n}")
    print(f"  position bias (fwd order prefers A): {n_pos_a_fwd}/{n} = {n_pos_a_fwd/n:.2f}")
    print(f"  position bias (rev order prefers A): {n_pos_a_rev}/{n} = {n_pos_a_rev/n:.2f}")
    print(f"  consistent pairs : {n_consistent}/{n} = {n_consistent/n:.2f}")
    if gold_a_count > 0:
        print(f"  accuracy vs gold : {correct_count}/{gold_a_count} = {correct_count/gold_a_count:.2f}  "
              f"(on consistent, non-tie pairs)")

    if n_consistent > 0:
        print(f"\n{'noise':>8}  {'flip_rate':>10}  {'invalids':>9}")
        print("-" * 35)
        results = []
        for scale in NOISE_SCALES[1:]:
            flip_rate = flip_counts[scale] / n_consistent
            results.append({"noise": scale, "flip_rate": flip_rate,
                             "invalids": invalid_counts[scale]})
            print(f"{scale:>8.3f}  {flip_rate:>10.3f}  {invalid_counts[scale]:>9}")
        return results


# ── mode: position-bias ───────────────────────────────────────────────────────

def run_position_bias_test(model, tokenizer, n_samples: int = 10):
    """
    Tests the robustness of position bias to input-embedding noise.

    Since the model consistently prefers position A regardless of content,
    we have a stable (if biased) baseline. We exploit this to run two
    parallel noise sweeps:

      Perturb A — add noise to the tokens of the answer in position A
                  (the one the model always picks). Measures how much noise
                  is needed to dislodge the position preference.

      Perturb B — add noise to the tokens of the answer in position B
                  (the one the model ignores). Pure control: if the model
                  truly ignores B's content, flip rate should stay near zero
                  at all noise levels.

    The asymmetry between the two curves shows whether any content processing
    is happening, and quantifies how robust the position bias is.

    Order is fixed as original (response_a in slot A) so the position bias
    is maximally expressed and results are reproducible.
    """
    csv_path = DATA_DIR / "helpsteer2_validation.csv"
    if not csv_path.exists():
        print(f"Data file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path).dropna(subset=["prompt", "response_a", "response_b"])
    df = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODE: POSITION-BIAS — noise robustness of position preference")
    print("=" * 70)
    print(f"Evaluating {len(df)} pairs in fixed A=response_a / B=response_b order.")
    print("Sweeping noise on position A (preferred) and position B (ignored).\n")

    flip_a = {s: 0 for s in NOISE_SCALES[1:]}   # noise on A → flipped to B
    flip_b = {s: 0 for s in NOISE_SCALES[1:]}   # noise on B → flipped to A
    inv_a  = {s: 0 for s in NOISE_SCALES[1:]}
    inv_b  = {s: 0 for s in NOISE_SCALES[1:]}
    n_valid = 0

    for i, (_, row) in enumerate(df.iterrows()):
        q  = str(row["prompt"])
        ra = str(row["response_a"])
        rb = str(row["response_b"])

        # Baseline: fixed order, no noise
        base_pref, base_raw = judge_preference(
            model, tokenizer, q, ra, rb, noise_scale=0.0, perturb_target="none",
        )

        print(f"\n{'─' * 70}")
        print(f"Pair {i+1}  gold={row.get('gold_label','?')}")
        print(f"  Q: {q[:120]!r}")
        print(f"  baseline  pref={base_pref}  raw={base_raw!r}")

        if base_pref not in ("A", "B"):
            print("  (no clear baseline — skipping)")
            continue
        n_valid += 1

        for scale in NOISE_SCALES[1:]:
            # noise on the preferred position
            pa_pref, pa_raw = judge_preference(
                model, tokenizer, q, ra, rb, noise_scale=scale, perturb_target="a",
            )
            # noise on the ignored position
            pb_pref, pb_raw = judge_preference(
                model, tokenizer, q, ra, rb, noise_scale=scale, perturb_target="b",
            )

            flip_a_flag = "FLIP" if pa_pref != base_pref and pa_pref in ("A","B") else "    "
            flip_b_flag = "FLIP" if pb_pref != base_pref and pb_pref in ("A","B") else "    "

            if pa_pref == "?":
                inv_a[scale] += 1
            elif pa_pref != base_pref:
                flip_a[scale] += 1

            if pb_pref == "?":
                inv_b[scale] += 1
            elif pb_pref != base_pref:
                flip_b[scale] += 1

            print(f"  noise={scale:.3f}  "
                  f"noiseA: pref={pa_pref} [{flip_a_flag}] raw={pa_raw!r}  |  "
                  f"noiseB: pref={pb_pref} [{flip_b_flag}] raw={pb_raw!r}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY  ({n_valid}/{len(df)} pairs had a clear baseline preference)\n")
    if n_valid == 0:
        return

    print(f"{'noise':>8}  {'flip_A':>8}  {'flip_B':>8}  {'diff':>8}")
    print("-" * 42)
    results = []
    for scale in NOISE_SCALES[1:]:
        fa = flip_a[scale] / n_valid
        fb = flip_b[scale] / n_valid
        print(f"{scale:>8.3f}  {fa:>8.3f}  {fb:>8.3f}  {fa-fb:>+8.3f}")
        results.append({"noise": scale, "flip_rate_A": fa, "flip_rate_B": fb})

    print("\n  flip_A: fraction of pairs where noise on position A changed the verdict")
    print("  flip_B: fraction of pairs where noise on position B changed the verdict")
    print("  diff  : asymmetry (large → model content-processes A but not B)")
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Perturbation sensitivity tests")
    parser.add_argument("--mode", choices=["decode", "judge", "position-bias", "both"],
                        default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model name or local path")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of Q/A pairs for judge/position-bias mode")
    parser.add_argument("--n-prompts", type=int, default=3,
                        help="Number of prompts for decode mode")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.mode in ("decode", "both"):
        run_decode_test(model, tokenizer, n_prompts=args.n_prompts)

    if args.mode in ("judge", "both"):
        run_judge_test(model, tokenizer, n_samples=args.n_samples)

    if args.mode == "position-bias":
        run_position_bias_test(model, tokenizer, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
