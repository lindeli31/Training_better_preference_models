# Gradient-Based Probing for Positional Bias in LLM Judges

## Motivation

Previous work using linear probes (diff_mean, PCA on contrastive differences) to find positional bias directions in LLM judge activations showed only weak causal attribution (+0.017 flip-rate difference over random baseline). The root cause: probing via random noise in a ~3072-dim space means <0.15% of perturbation energy lands in the 3-dim bias subspace — the model never actually experiences a meaningful push along the bias direction.

This document proposes a gradient-based alternative, adapting the activation steering methodology from [\[TODO: cite paper\]](https://arxiv.org/html/2506.16078v1) to the positional bias setting.

---

## Core Idea: Gradient as Causal Probe

Instead of finding a bias direction statistically (mean difference between conditions), we find it causally — by asking the model's own computation graph where it is sensitive.

For each evaluation prompt where the model picks response A (first position):

```
g_l = ∇_{h_l} NLL(verdict=B)
```

`g_l` is the direction in activation space at layer `l` that, if moved along, would most directly increase the probability of verdict B. This is the causally relevant direction for verdict switching — by construction.

**Key advantage over diff_mean**: diff_mean finds where activations *statistically differ* between conditions. The gradient finds where activations *causally drive* the output. These are not the same thing.
20.250.13.243
---

## Experiment 1: Layer-Wise Sensitivity Profile (LASR-style)

Run the gradient-based perturbation at every layer and record how much the verdict moves.

**Procedure**:
1. For each prompt, for each layer `l`:
   - Forward pass → capture `h_l`
   - Backward pass → compute `g_l = ∇_{h_l} NLL(B)`
   - Perturb: `h_l' = h_l + α · g_l / ||g_l||`
   - Complete forward pass from layer `l` onward → record ΔNLL(B) and verdict change
2. Average ΔNLL across prompts per layer → **layer-wise sensitivity curve**

**Expected shapes and interpretations**:

| Shape | Interpretation |
|---|---|
| Sharp mid-layer peak (~15–22) | Bias encoded in a bottleneck; one intervention point; correctable |
| Late-layer peak (~25–31) | Correct content representation, position overrides at decision stage |
| Flat / distributed | Bias is structural across all layers; no clean intervention point |

---

## Experiment 2: Swap Antisymmetry — Is the Gradient a Pure Position Signal?

This is the key experiment to establish whether the gradient captures **position** or **content**.

**Setup**: For paired prompts (A,B) and their swap (B,A) with the same underlying responses:

- On (A,B) prompt: compute `g_AB = ∇_h NLL(B)` — gradient steering toward the second response
- On (B,A) prompt: compute `g_BA = ∇_h NLL(A)` — gradient steering toward the second response (now A)

**Measure**: `cosine(g_AB, -g_BA)` at the peak-sensitivity layer.

| Value | Interpretation |
|---|---|
| ≈ 1.0 | Pure position signal: both gradients point along the same axis regardless of content |
| ≈ 0.5 | Mixed signal: gradient captures both position and content quality |
| ≈ 0.0 | Content-driven: gradient tracks response quality, not position |

---

## Experiment 3: Antisymmetry as a Classifier for Positional Bias

**This is the novel contribution**: use the swap antisymmetry score itself as a *detector* of whether a given pair exhibits positional bias.

**Theory**:

- **Non-positionally-biased pairs** (model judgment tracks content): `cosine(g_AB, -g_BA) ≈ 1.0`
  The model's sensitivity to verdict switching is driven by position in both directions — it would flip for positional reasons symmetrically. The gradient is a pure position signal even when the model is *not* being biased, because the positional axis is always present — it's just not what's determining the verdict.

- **Positionally biased pairs** (model picks A because it's first, not because it's better): `cosine(g_AB, -g_BA)` deviates from 1.0 (closer to 0.0)
  The gradient for biased pairs is contaminated by content — because the model's "pressure" toward the correct answer also incorporates the quality signal it's suppressing. The position axis and the quality axis are entangled.

**Implication**: A pair where the model's verdict is positionally biased should show a *lower* antisymmetry score than a pair where the model judges correctly. We can operationalize this:

1. Label pairs as positionally biased using the swap test (model picks A on (A,B) but picks A on (B,A) too → biased toward A regardless)
2. Compute antisymmetry scores for biased vs. unbiased pairs
3. Test whether antisymmetry is a reliable signal → receiver operating characteristic (ROC) for positional bias detection

If this holds, the gradient antisymmetry score is a **probe-free, input-agnostic detector** of positional bias at the activation level.

---

## Experiment 4: Causal Intervention — Dose-Response

At the peak-sensitivity layer, vary the perturbation scale `α` and measure:

- `P(verdict flips A→B | steer along g_AB)` vs. `α`  
- `P(verdict flips A→B | steer along random direction)` vs. `α` (baseline)
- ΔNLL(B) vs. `α`

**Expected result if gradient captures causal position mechanism**:
```
flip rate
    |         gradient direction
    |        ___________
    |       /
    |______/              _____ random baseline (flat)
    0    0.1   0.5   1.0   α
```

A steep, smooth dose-response curve along the gradient direction that is absent for random directions establishes that the gradient is finding a causally specific — not merely energetically large — direction.

**Interpretive question for the flip**: does the post-intervention verdict reflect corrected quality judgment, or just reversed positional bias? Disambiguate using human-annotated ground-truth preference labels:
- If post-intervention verdict matches human preference more often → intervention corrected the bias
- If post-intervention verdict just tracks "whichever was second" → bias reversed, not corrected

---

## Experiment 5: Comparing Posttrained Models

Run the full pipeline (layer sensitivity, antisymmetry, dose-response) across models that differ in posttraining:

- Base model (no RLHF/instruction tuning)
- Instruction-tuned (SFT only)
- RLHF-trained / preference-optimized

**Hypothesis**: Posttraining sharpens the position bias signal — the gradient becomes more concentrated at a specific layer and the antisymmetry score becomes more pronounced — because preference optimization reinforces whatever heuristics the model uses to produce consistent verdicts, including positional ones.

Concretely, we'd expect:
- Base model: flat layer-sensitivity curve, low antisymmetry
- RLHF model: peaked layer-sensitivity curve, high antisymmetry on unbiased pairs

This would provide mechanistic evidence that **RLHF amplifies positional bias** as a side effect of training for verdict consistency. (Coralie annotation: wrong. We'd expect that even if positional bias is still encoded in latent representations, the model is less sensitive to perturbations and also less likely to get into "toxic" positional bias subspace. Depending on how a model is finetuned, these latent representations of postional bias might not exist at all. If they do exist, we expect they are sitll perturbationally sensitive)

---

## Summary of Metrics

| Metric | What it measures |
|---|---|
| Layer-wise ΔNLL | Where in the model positional bias is causally encoded |
| `cosine(g_AB, -g_BA)` | Whether the gradient captures position vs. content |
| ROC(antisymmetry → bias label) | Whether antisymmetry detects positionally biased pairs |
| Dose-response slope | Causal specificity of the gradient direction |
| Post-intervention accuracy vs. human labels | Whether intervention corrects or reverses bias |
| Cross-model antisymmetry comparison | Whether posttraining amplifies positional bias |

---

## Why This Is Better Than What We Had

| Issue with previous approach | How gradient probing fixes it |
|---|---|
| 3 bias directions in 3072-dim space → <0.15% noise energy in bias subspace | Gradient moves exactly along the causally relevant direction; no dimensionality dilution |
| diff_mean is correlational (statistical fingerprint) | Gradient is causal (defined by the computation graph) |
| Binary flip rate is low-sensitivity | ΔNLL is continuous; detects signal before verdict crosses threshold |
| Fixed layer (layer 20) | Layer sweep finds the actual locus of position encoding |
| No way to distinguish "position" from "content" directions | Swap antisymmetry cleanly separates the two |
