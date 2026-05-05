# Perturbation Sensitivity

Tests whether injecting Gaussian noise into an LLM's internal representations
changes what it outputs, and in particular whether it changes an LLM judge's
preference between two answers.

The two modes use **different noise injection points** for good reasons —
see below.

## Setup

Uses the base `conda` environment (Python 3.11, torch 2.2, transformers 4.37).
Default model: `meta-llama/Llama-3.2-1B-Instruct` (~1B params, ~2 GB in float16,
loads comfortably on 16 GB Apple M1 via MPS).

Llama models require a HuggingFace account with Llama access. Log in once before
running:

```bash
huggingface-cli login
```

Then run:

```bash
/path/to/anaconda3/bin/python perturb_hidden.py [--mode decode|judge|both]
```

## Modes

### Decode mode — last-hidden-layer hook

```bash
python perturb_hidden.py --mode decode [--n-prompts 3]
```

A `register_forward_hook` is placed on the last transformer block. On every
forward call during generation (once per new token), Gaussian noise is added
to the entire hidden state tensor of shape `[batch, seq_len, hidden_dim]`.

**Why the last hidden layer here?** For decode mode the question is simply
"does the model still produce coherent text when its final representations are
perturbed?" There is no need to target a specific part of the input, so
whole-sequence noise is the right thing to measure.

The hook is registered before `generate()` and removed immediately after,
so it has no side effects on subsequent calls.

### Judge mode — input token embedding perturbation

```bash
python perturb_hidden.py --mode judge [--n-samples 10]
```

The question being tested is: *if we perturb the representation of the answer
the judge originally preferred, does it still prefer it?*

To do that surgically we need noise that originates from the target answer's
tokens, not from everything else in the prompt. Here is how it works:

**Step 1 — Locate the answer's token span**

The full judge prompt is a single string:
```
... Question: {q} ... Answer A: {a} ... Answer B: {b} ...
```
The tokenizer's offset mapping gives us the `(char_start, char_end)` character
span for every token in the sequence. We use this to find exactly which token
indices correspond to the text of answer A (or B), avoiding the
context-dependent tokenization issues that arise when tokenizing substrings
independently.

**Step 2 — Add noise at the embedding level**

Every token ID is first converted to an embedding vector via the embedding
matrix (`model.get_input_embeddings()`). This happens before any attention
computation. We call this lookup for the full prompt, then add Gaussian noise
exclusively to the embedding vectors at the target answer's token positions:

```
token_ids  →  embed_layer(token_ids)  →  add noise at positions [tok_start:tok_end]
                                                ↓
                                        forward pass (all layers, attention, etc.)
                                                ↓
                                        model generates "A" or "B"
```

The model then receives `inputs_embeds` instead of `input_ids`.

**Why this is cleaner than a last-layer hook for judge mode**

A hook on the last hidden layer operates on states that are already a mixture
of all tokens (self-attention has run through all layers by then). There is no
way to say "only add noise to the part of the last hidden state that came from
answer A" — that information is entangled with everything else.

At the embedding level, before any attention, each position still corresponds
exactly to one answer's token. The noise originates cleanly from those positions.

**Known limitation**

Self-attention propagates information across all positions at every layer.
After the first attention layer, the noise from answer A's positions has
influenced B's hidden states and vice versa. By the final layer the
perturbation is not perfectly isolated. This is unavoidable in a full-attention
architecture without modifying the attention mask.

**What the experiment measures**

1. **Baseline**: ask the judge with no noise → record preferred answer (A or B).
2. **Sweep**: for increasing noise scales σ, re-run with noise on the preferred
   answer's embeddings → record whether the judge flips its preference.
3. **Flip rate**: fraction of pairs where the preference changed at each σ.

A low flip rate at small σ means the judge's preference is robust to small
perturbations of the preferred answer's representation. A rapidly rising flip
rate suggests the preference is fragile.

## Noise scales

Both modes sweep `σ ∈ {0, 0.01, 0.05, 0.1, 0.3, 1.0}`. The token embeddings
in GPT-2 are roughly O(1) in magnitude after the embedding lookup, so these
scales are interpretable as fractions of the typical embedding norm.

## Data

Judge mode loads Q/A pairs from `../data/helpsteer2_validation.csv`
(columns: `prompt`, `response_a`, `response_b`).
