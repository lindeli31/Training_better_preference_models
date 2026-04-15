# LLM Judge Position Bias: GEPA Optimisation Results

## Experimental Setup

### Task
An LLM judge is given two candidate responses to a conversation and must pick which is better (`a`, `b`, or `equal`). The core question is whether the judge's verdict changes depending on the *order* the responses are presented — a known failure mode called **position bias**.

### Model
- **Judge / Task LM**: `meta-llama/Llama-3.3-70B-Instruct` (temperature 1.0, no cache)
- **Reflection LM** (used by GEPA to evolve prompts): same model
- **Serving stack**: SwissAI / CSCS OpenAI-compatible endpoint

### Dataset
- **Source**: [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3), `preference` split, `train` partition (38 459 raw examples)
- **Filter**: total length of conversation + both responses ≤ 1 000 characters → **6 799 examples** pass
- **Splits** (matching GEPA paper proportions for some datasets (HotPotQA, etc.)):

| Split | Size | Index range |
|---|---|---|
| Train | 150 | 0–149 |
| Validation | 300 | 150–449 |
| Test (held-out) | 300 | 450–749 |

### Prompt Template
Each example is run **twice** — once in AB order and once in BA order — via `BatchPermutationWrapper`. Each call uses the following DSPy `Predict` signature (before GEPA evolves it):

```
System:
  Evaluate which of two responses better answers the conversation.
  Your judgment must be consistent regardless of the order the responses are presented.

Input fields:
  answer_candidate_a  — First possible answer
  answer_candidate_b  — Second possible answer
  conversation_history — Messages in conversation history

Output field:
  preference — the better answer: a, b, or 'equal'
```

GEPA is free to rewrite the system docstring during optimisation.

### Optimiser: GEPA
- **Algorithm**: GEPA (Reflective Prompt Evolution, ICLR 2026) — evolves the system prompt via reflection over metric feedback
- **Setting**: `auto="heavy"` → 18 full evaluation passes over the validation set
- **Metric optimised**: arithmetic mean of `accuracy_score` and `consistency_score` (see definitions below)
- **Feedback to GEPA per rollout**: combined score + per-ordering accuracy breakdown + consistency score

---

## Metric Definitions

### Accuracy (both perms)
Average accuracy across both orderings (AB and BA). For each ordering, the judge's answer is compared to the ground-truth preference, accounting for label remapping (when inputs are swapped, the correct answer label also swaps). A score of 1.0 means the judge got the right answer in both orderings; 0.5 means one of two correct.

### Accuracy (AB only)
Accuracy on the AB ordering alone (response 1 in slot A, response 2 in slot B). Sensitive to position bias: a model with strong primacy bias will score high here but low on BA.

### Debiased accuracy
`0.5 × acc_AB + 0.5 × acc_BA_remapped`. Averages accuracy across both orderings after remapping BA labels back to the original reference frame. Robust to position bias — a model that always picks `a` regardless of content scores ~0.5 here.

### Flip rate
Fraction of examples where the judge gives *different* answers for AB vs BA (after label remapping). A perfectly consistent judge scores 0.0. A perfectly inconsistent judge scores 1.0. Note: lower flip rate is better, but very low flip rate combined with low accuracy means the model is consistently *wrong*.

### Consistency score
1.0 if the judge correctly flips its answer when inputs are swapped (i.e., picks `a` for AB and `b` for BA, or vice versa), 1.0 if both orderings yield `equal`. 0.0 otherwise. Captures whether the model understands the symmetry of the task. Higher is better.

### Combined (acc + cons)
Arithmetic mean of `accuracy_score` (both perms) and `consistency_score`. This is the metric GEPA directly optimises. Balances being right with being stable.

### Position bias metrics
| Metric | Meaning |
|---|---|
| Inconsistent primacy rate | Rate at which model picked first-presented in both orderings |
| Inconsistent recency rate | Rate at which model picked last-presented in both orderings |
| Equal rate | Fraction of examples where model outputs `equal` in AB ordering. |
| Acc when gold in A | Accuracy when the ground-truth better response is in slot A. |
| Acc when gold in B | Accuracy when the ground-truth better response is in slot B. |
| Position gap (A − B) | Difference between the two above. 0 = unbiased; positive = model favours whatever is in slot A. |

---

## Results

### Main Metrics

| Metric | Baseline (Val) | Baseline (Test, 99% CI) | After GEPA (Val) | After GEPA (Test, 99% CI) |
|---|---|---|---|---|
| Accuracy (both perms) | 0.342 | 0.313 [0.258, 0.370] | 0.387 | 0.317 [0.252, 0.375] |
| Accuracy (AB only) | 0.330 | 0.300 [0.237, 0.363] | 0.367 | 0.310 [0.243, 0.377] |
| Debiased accuracy | 0.342 | 0.313 [0.258, 0.370] | 0.387 | 0.317 [0.252, 0.375] |
| Flip rate | 0.343 | 0.370 [0.307, 0.440] | 0.273 | 0.280 [0.317, 0.390] |
| Consistency | 0.577 | 0.590 [0.520, 0.657] | 0.690 | 0.663 [0.587, 0.730] |
| **Combined (acc+cons)** | **0.459** | **0.452 [0.411, 0.494]** | **0.538** | **0.490 [0.449, 0.533]** |

Test CIs are 99% percentile bootstrap (1 000 resamples, seed 42).

### Position Bias

| Metric | Baseline (Val) | Baseline (Test) | After GEPA (Val) | After GEPA (Test) |
|---|---|---|---|---|
| Inconsistent primacy rate | 0.207 | 0.233 | 0.183 | 0.197 |
| Inconsistent recency rate | 0.067 | 0.050 | 0.070 | 0.090 |
| Equal rate | 0.103 | 0.087 | 0.033 | 0.043 |
| Acc when gold in A | 0.388 | 0.385 | 0.388 | 0.338 |
| Acc when gold in B | 0.283 | 0.224 | 0.352 | 0.293 |
| Position gap (A − B) | +0.104 | +0.161 | −0.036 | +0.045 |

---

