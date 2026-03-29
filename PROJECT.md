# Project: Training Better Preference Models

## Overview

This project investigates **systematic biases and sensitivities in LLM-as-a-judge evaluations**. The core question: when we use a language model to judge which of two responses is better, how much does the verdict depend on factors *other than response quality* — like presentation order, prompt wording, or reasoning depth?

**Model**: Qwen/Qwen3-30B-A3B-Instruct-2507
**Inference backend**: Swiss AI Stack (`https://serving.swissai.svc.cscs.ch/v1`) — OpenAI-compatible API, no local cluster/Slurm/vLLM
**Datasets**: nvidia/HelpSteer2 (default), Anthropic/hh-rlhf

---

## Research Questions

1. **Position Bias (B1)**: Does the judge prefer whichever response appears first (or second), regardless of quality?
2. **Template Sensitivity (B2)**: Does changing the system prompt framing ("expert rater" vs "LLM judge" vs "neutral" etc.) change the verdict?
3. **Reasoning Depth (B3)**: Does explicitly asking the model to reason before judging (prompt-elicited Chain of Thought) improve accuracy against human gold labels?
4. **Input Sensitivity (B4)**: Do semantically equivalent paraphrases of the same prompt produce different verdicts? Does changing the evaluation criterion (helpful vs accurate vs concise etc.) drift the labels?
5. **User-Prompt Structure (B5)**: Do structural properties of the *user prompt* — specifically response label type (A/B vs 1/2) and criterion placement (before vs after responses) — affect judge verdicts independently of system-prompt wording?

---

## Architecture

```
HuggingFace datasets
        │
        ▼
   src/dataset.py          ──→  list[PairRecord]
        │
        ▼
  src/experiments.py        ──→  builds call dicts per experiment conditions
        │
        ▼
 src/inference_client.py    ──→  async HTTP POST to Swiss AI Stack
        │                         (aiohttp, semaphore concurrency, exponential backoff)
        ▼
   Swiss AI Stack           ──→  /v1/chat/completions (Qwen3-30B-A3B)
        │
        ▼
   Label extraction         ──→  regex cascade: verdict lines → bare label → fallback
        │
        ▼
   JSONL results            ──→  results/<experiment>/*.jsonl
        │
        ▼
    src/metrics.py          ──→  position consistency, pairwise agreement, accuracy vs gold
        │
        ▼
  run_experiments.py        ──→  CLI orchestrator, prints metric summaries
```

---

## File-by-File Details

### `src/inference_client.py`
- **InferenceConfig** dataclass: base_url, api_key, model, generation params (max_tokens=2048, temperature=0.0, top_p=1.0), retry config (max_retries=5, base_delay=2s, max_delay=60s), concurrent_requests=8
- **JudgeResponse** dataclass: prompt_id, experiment_id, condition, raw_text, label (A/B/C/None), thinking (extracted CoT), latency_s, total_tokens, parse_ok, error
- **Label extraction**: Three-tier regex cascade:
  1. Verdict lines: `(?i)(?:verdict|answer|response|output|better response)\s*:\s*([ABC])\b`
  2. Bare label at end: `(?i)\b([ABC])\s*$`
  3. Label + punctuation at end: `(?i)\b([ABC])[.\s]*\Z`
  4. Fallback: last uppercase `\b([ABC])\b` in text (case-sensitive to avoid matching article "a")
  - `<think>...</think>` blocks are stripped before extraction
- **SwissAIClient**: async context manager with aiohttp.ClientSession
  - `_post()`: sends to `/chat/completions`, handles 429 with exponential backoff, raises RuntimeError if all retries exhausted
  - `judge()`: builds messages, adds thinking/CoT extra_body params if budget > 0, parses response
  - `batch_judge()`: dispatches list of calls via `asyncio.gather`
  - Semaphore controls max concurrent requests

### `src/templates.py`
- **System prompt variants** (12 total):
  - `expert_rater` — baseline "You are an expert rater" (used in B1, B2, B3, B4, B5)
  - `llm_judge` — "You are an LLM used to judge" (B2)
  - `neutral` — imperative, no persona (B2)
  - `academic` — formal quality evaluator (B2)
  - `minimal` — one-liner (B2)
  - `reason_then_judge` — explain reasoning about strengths/weaknesses, then verdict (B3)
  - `structured_reasoning` — rate on helpfulness/accuracy/coherence criteria, then verdict (B3)
  - `expert_rater_alt1/alt2/alt3` — minor wording variants of expert_rater (B4)
  - `blind` — same body as expert_rater but 1/2 output labels; user prompt uses [Response 1]/[Response 2] (B5)
  - `criterion_first` — A/B labels; criterion question placed before responses in user prompt (B5)
  - `blind_criterion_first` — 1/2 labels; criterion question before responses (B5)
- **Criteria**: helpful, quality, accurate, harmless, concise — mapped to natural language fragments
- **build_prompt()**: resolves template_id + criterion → (system_prompt, user_prompt)
- **User prompt formats** (B5 adds three structural variants):
  - Standard (B1–B4): `[User Prompt]\n{prompt}\n\n[Response A]\n{a}\n\n[Response B]\n{b}\n\nWhich response is {criterion}?`
  - Blind: same structure with `[Response 1]`/`[Response 2]` labels
  - Criterion-first: criterion question before `[Response A]`/`[Response B]`
  - Blind criterion-first: criterion question before `[Response 1]`/`[Response 2]`

### `src/dataset.py`
- **PairRecord** dataclass: prompt_id, prompt, response_a, response_b, gold_label, plus five optional stratification fields:
  - `score_gap` (float): |score_a - score_b|
  - `difficulty` (str): "easy" / "medium" / "hard" derived from score_gap thresholds
  - `verbosity_delta` (int): verbosity_a - verbosity_b (signed; positive = better response is more verbose)
  - `complexity_max` (int): max(complexity_a, complexity_b), scale 0–4
  - `is_multiturn` (bool): True if prompt contains a prior `Assistant:` turn
  - `flipped()` returns new PairRecord with A↔B swapped, gold_label adjusted (A↔B, C→C), verbosity_delta negated; symmetric fields (score_gap, difficulty, complexity_max, is_multiturn) preserved unchanged
- **`_detect_multiturn(prompt)`**: regex check for `\bassistant\s*:` to identify multi-turn conversations
- **`_stratified_sample(pairs, n, rng)`**: sample n pairs proportionally across difficulty strata with overflow handling
- **`download_dataset()`**: Groups by prompt, takes best/worst by mean(helpfulness + correctness + coherence)/3. Gold label "A" (best always in position A) or "C" for ties. Now always writes all five stratification fields to the standard file. Full mode additionally writes per-response attribute scores and response lengths.
- **`load_dataset_pairs(split, n, seed, stratify, randomize_position)`**:
  - `stratify=True`: proportional sampling across difficulty strata (prevents easy-pair oversampling); falls back to random sampling with warning if difficulty metadata is absent
  - `randomize_position=True`: randomly flips 50% of pairs so gold_label is ~50% A / ~50% B — **required for valid positional bias measurement** (without it, gold is always A and A-preference is indistinguishable from accuracy)

### `src/experiments.py`
- **B1 `run_position_bias()`**: For each pair, creates two calls — condition "AB" (original order) and "BA" (flipped order via `pair.flipped()`). The `prompt_id` stays the same for both so metrics can match them.
- **B2 `run_template_sensitivity()`**: Judges each pair with 5 template variants (expert_rater, llm_judge, neutral, academic, minimal). Same order, same data.
- **B3 `run_reasoning_depth()`**: 3 prompt-elicited reasoning conditions:
  - `no_reasoning`: expert_rater template — just output A, B, or C
  - `reason_then_judge`: explain reasoning about strengths/weaknesses, then give verdict
  - `structured_reasoning`: rate each response on helpfulness/accuracy/coherence, then give verdict
- **B4 `run_input_sensitivity()`**: 4 template variants × 5 criteria = 20 conditions per pair
- **B5 `run_user_prompt_structure()`**: 2×2 factorial — label type (A/B vs 1/2) × criterion position (after/before). Each condition run in both AB and BA order → 8 calls per pair. Condition names: `{template}_AB` / `{template}_BA`. Enables per-condition position bias measurement.
- All experiments save results to JSONL via `save_jsonl()`

### `src/metrics.py`
- **`compute_position_bias()`**: Groups AB/BA by prompt_id. Consistent = flipped label matches (A↔B). Tracks bias toward first/second position, tie inconsistency.
- **`compute_pairwise_agreement()`**: Used for B2 and B4. Pivots by prompt_id × condition, computes pairwise agreement rate, identifies most volatile pairs, tracks label distribution per condition.
- **`compute_thinking_accuracy()`**: Used for B3 and B5 (accuracy per structural condition). Computes accuracy vs gold labels per condition, agreement vs no_thinking baseline, average latency per condition.
- **`compute_stratified_metrics(results, pairs, stratum_key, base_metric_fn, **kwargs)`**: Partition results by any stratification field on PairRecord and apply any base metric function separately to each stratum. Returns `{stratum_value: metrics_dict}`. Used to report position bias per difficulty bucket, per verbosity direction, per turn count, etc.
- **`print_summary()`**: Pretty-prints metric dicts to console.

### `run_experiments.py`
- CLI entry point with argparse
- Loads API key from `SWISSAI_API_KEY` env var
- Runs selected experiments (or all), computes metrics, prints summaries
- Key CLI flags: `--experiments`, `--n-pairs`, `--dataset`, `--split`, `--criterion`, `--template`, `--concurrency`, `--output-dir`, `--seed`
- Experiments: `position_bias`, `template_sensitivity`, `reasoning_depth`, `input_sensitivity`, `user_prompt_structure`
- `--stratify`: proportional difficulty sampling; auto-reports B1 by stratum when metadata present
- `--randomize-position`: 50/50 gold label distribution for valid positional bias measurement

### `tests/test_pipeline.py`
- 39 unit tests, no API calls required
- Tests: label extraction (bare, verdict line, thinking blocks, fallback, blind 1/2 labels), template building (B5 variants), PairRecord metadata fields and flipping behaviour, multi-turn detection, stratified sampling (proportional, undersized stratum, overflow), randomize_position (gold label distribution, response identity, verbosity_delta sign), all three base metric functions, compute_stratified_metrics (by difficulty, None-skipping, BA-condition matching)
- Can run with `python3 tests/test_pipeline.py` or `pytest tests/`

---

## Key Design Decisions

1. **Pure HTTP client** — no local model serving. All inference goes through the Swiss AI Stack's OpenAI-compatible API. This keeps the codebase simple and portable.

2. **Prompt-elicited reasoning only** — B3 uses three levels of prompt-elicited reasoning (no reasoning, free-form reasoning, structured criteria-based reasoning). All reasoning depth is controlled entirely by the prompt template — there is no native API thinking-budget feature involved. This keeps the design simple and portable across any OpenAI-compatible endpoint.

3. **Label extraction cascade** — the regex tries explicit verdict lines first (most reliable), then bare labels at end of text, then falls back to the last uppercase A/B/C in the text. `<think>` blocks are stripped before extraction to avoid matching reasoning content. Labels `1` and `2` (from B5 blind templates) are matched by the end-anchored patterns and immediately normalised to A/B, keeping all downstream metric functions compatible.

4. **Position bias uses same prompt_id** — both AB and BA conditions for the same pair share the original `prompt_id`. The `flipped()` method's `prompt_id + "_flipped"` is NOT used as the call's prompt_id. This lets `compute_position_bias()` match pairs by prompt_id.

7. **Gold label always A is a known confound** — `download_dataset()` always places the better response in position A (gold="A") except for ties. This means without `randomize_position=True`, any judge A-preference looks like accuracy. The `--randomize-position` flag fixes this by randomly placing the better response in either position at load time.

8. **Dataset heterogeneity controlled via stratification** — `--stratify` ensures proportional representation of easy/medium/hard pairs. Without it, random sampling is dominated by easy pairs where quality differences are unambiguous and positional bias is minimal — the stress-test cases (hard pairs) are systematically underrepresented.

5. **Semaphore-based concurrency** — `asyncio.Semaphore(8)` limits concurrent requests to avoid overwhelming the API. All calls in an experiment are dispatched via `asyncio.gather` for maximum parallelism within the limit.

6. **Exponential backoff** — on 429 or ClientError, waits `base_delay * 2^attempt` seconds (capped at 60s). Raises RuntimeError if all 5 retries exhausted.

---

## B5 Pilot Results (n=10, criterion=helpful, 2026-03-29)

Smoke test on 10 validation pairs. Results are preliminary; full run at n=200 required.

**Position consistency per structural condition:**

| Condition | Pos. consistency | Bias 1st | Bias 2nd | Acc. vs gold |
|---|---|---|---|---|
| `expert_rater` (baseline) | 0.90 | 0.10 | 0.00 | 0.50 |
| `blind` | 0.80 | 0.00 | 0.20 | 0.40 |
| `criterion_first` | 0.60 | 0.20 | 0.20 | 0.50 |
| `blind_criterion_first` | 0.70 | 0.00 | 0.30 | 0.40 |

**Cross-condition pairwise agreement (AB order only):** 0.817

**Label distribution (AB order only):**

| Condition | A | B | C |
|---|---|---|---|
| `expert_rater` | 7 | 3 | 0 |
| `blind` | 4 | 6 | 0 |
| `criterion_first` | 6 | 4 | 0 |
| `blind_criterion_first` | 4 | 6 | 0 |

**Key observations:**
- Blind labels (1/2) **reverse the direction of positional bias**: the A-preference (70% with A/B labels) disappears entirely and becomes a B/second-position preference (60%), suggesting a learned A-letter bias in the model.
- Criterion-first placement **increased** positional instability (0.60 vs 0.90), contrary to the hypothesis. Pre-framing may cause early anchoring that makes the judge more sensitive to response reordering.
- Cross-condition agreement of 0.817 means ~18% of pairs get different labels from user prompt structure alone, with identical system prompts and data.
- Accuracy figures (0.40–0.50) are inconclusive at n=10.

---

## Bugs Fixed (2026-03-26)

1. **`_post()` silent None return** (`inference_client.py`): If all retries exhausted on HTTP 429, the function returned None implicitly, causing TypeError downstream. Fixed by adding `raise RuntimeError` after the retry loop.

2. **Case-insensitive label patterns** (`inference_client.py`): Patterns 2 and 3 only matched uppercase [ABC]. Added `(?i)` flag. The fallback remains case-sensitive to avoid matching the English article "a".

3. **Tautological test** (`test_pipeline.py`): `test_extract_label_fallback` had `assert label is None or ok is not True or label in ("A", "B", "C")` which is always True. Fixed to `assert label is None and ok is False`.

---

## Output Format

JSONL files in `results/<experiment>/`. Each line:

```json
{
  "prompt_id": "a3f9b12c10",
  "experiment_id": "position_bias",
  "condition": "AB",
  "raw_text": "Response A is more helpful...\nVerdict: A",
  "label": "A",
  "thinking": null,
  "latency_s": 1.234,
  "total_tokens": 512,
  "parse_ok": true,
  "error": null
}
```

---

## How to Run

```bash
# Setup
pip install -r requirements.txt
export SWISSAI_API_KEY="<key>"

# Download dataset (one-time; generates stratification metadata)
python -m src.dataset

# All experiments (200 pairs)
python3 run_experiments.py

# Recommended: stratified + randomized position for valid bias measurement
python3 run_experiments.py --stratify --randomize-position --n-pairs 200

# Single experiment with stratification breakdown
python3 run_experiments.py --experiments position_bias --stratify --randomize-position

# Quick smoke test
python3 run_experiments.py --n-pairs 20 --experiments position_bias

# Tests (no API key needed)
python3 tests/test_pipeline.py
```

---

## Project Status

- [x] Core pipeline implemented (inference client, templates, dataset loaders, experiments, metrics)
- [x] B5 experiment (user-prompt structure: blind labels + criterion position)
- [x] Dataset stratification: 5 metadata fields, stratified sampling, randomize_position, compute_stratified_metrics
- [x] Unit tests (39/39 passing)
- [x] Architecture document (architecture.pdf) — updated to include B5 and pilot results
- [x] README
- [x] Bug fixes and code review
- [x] B5 smoke test (n=10, criterion=helpful) — results in "B5 Pilot Results" section above
- [ ] Full B5 run (n=200)
- [ ] Run B1–B4 experiments on Swiss AI Stack
- [ ] Analysis and visualization of results
- [ ] Paper/report writing
