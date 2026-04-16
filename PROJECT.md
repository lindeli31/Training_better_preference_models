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
- **System prompt variants** (9 total):
  - `expert_rater` — baseline "You are an expert rater" (used in B1, B2, B3, B4)
  - `llm_judge` — "You are an LLM used to judge" (B2)
  - `neutral` — imperative, no persona (B2)
  - `academic` — formal quality evaluator (B2)
  - `minimal` — one-liner (B2)
  - `reason_then_judge` — explain reasoning about strengths/weaknesses, then verdict (B3)
  - `structured_reasoning` — rate on helpfulness/accuracy/coherence criteria, then verdict (B3)
  - `expert_rater_alt1/alt2/alt3` — minor wording variants of expert_rater (B4)
- **Criteria**: helpful, quality, accurate, harmless, concise — mapped to natural language fragments
- **build_prompt()**: resolves template_id + criterion → (system_prompt, user_prompt)
- **User prompt format**: `[User Prompt]\n{prompt}\n\n[Response A]\n{a}\n\n[Response B]\n{b}\n\nWhich response is {criterion}?`

### `src/dataset.py`
- **`DIFFICULTY_LEVELS`**: module-level constant `("easy", "medium", "hard")` — single source of truth used by both the loader and the CLI.
- **PairRecord** dataclass: prompt_id, prompt, response_a, response_b, gold_label, difficulty (optional — `None` when loaded from the standard dataset, `"easy"/"medium"/"hard"` when loaded from the full variant)
  - `flipped()` returns new PairRecord with A↔B swapped, gold_label adjusted (A↔B, C→C), difficulty propagated unchanged
- **HelpSteer2 loader**: scored dataset (not pairwise). Groups by prompt, takes best/worst by mean(helpfulness + correctness + coherence)/3. Gold label derived from score comparison. Full variant adds per-response scores, score_gap, lengths, and difficulty tag (easy: gap > 1.0, medium: gap > 0.33, hard: gap ≤ 0.33).
- **HH-RLHF loader**: pairwise dataset. Extracts last assistant turn from chosen/rejected. Gold label always "A" (chosen is preferred).
- **load_dataset_pairs()**: loads all, shuffles with seed, truncates to n pairs. Accepts optional `difficulty` filter — when set, automatically loads the `_full` JSON variant and filters before sampling.

### `src/experiments.py`
- **B1 `run_position_bias()`**: For each pair, creates two calls — condition "AB" (original order) and "BA" (flipped order via `pair.flipped()`). The `prompt_id` stays the same for both so metrics can match them.
- **B2 `run_template_sensitivity()`**: Judges each pair with 5 template variants (expert_rater, llm_judge, neutral, academic, minimal). Same order, same data.
- **B3 `run_reasoning_depth()`**: 3 prompt-elicited reasoning conditions:
  - `no_reasoning`: expert_rater template — just output A, B, or C
  - `reason_then_judge`: explain reasoning about strengths/weaknesses, then give verdict
  - `structured_reasoning`: rate each response on helpfulness/accuracy/coherence, then give verdict
- **B4 `run_input_sensitivity()`**: 4 template variants × 5 criteria = 20 conditions per pair
- All experiments save results to JSONL via `save_jsonl()`

### `src/metrics.py`
- **`compute_position_bias(results, gold_labels=None)`**: Groups AB/BA by prompt_id. Consistent = flipped label matches (A↔B). Tracks bias toward first/second position, tie inconsistency. If `gold_labels` is provided, also computes `ab_accuracy`, `ba_accuracy`, `overall_accuracy`, and `accuracy_gap` (AB − BA). A large gap indicates the judge is picking correctly in AB partly due to position preference rather than quality recognition.
- **`compute_pairwise_agreement()`**: Used for B2 and B4. Pivots by prompt_id × condition, computes pairwise agreement rate, identifies most volatile pairs, tracks label distribution per condition.
- **`compute_thinking_accuracy()`**: Used for B3. Computes accuracy vs gold labels per condition, agreement vs no_thinking baseline, average latency per condition.
- **`print_summary()`**: Pretty-prints metric dicts to console.

### `run_experiments.py`
- CLI entry point with argparse
- Loads API key from `SWISSAI_API_KEY` env var
- Runs selected experiments (or all), computes metrics, prints summaries
- Key CLI flags: `--experiments`, `--n-pairs`, `--dataset`, `--split`, `--criterion`, `--template`, `--difficulty`, `--concurrency`, `--output-dir`, `--seed`

### `tests/test_pipeline.py`
- 15 unit tests, no API calls required
- Tests: label extraction (bare, verdict line, thinking blocks, fallback), template building, PairRecord flipping, all three metric functions
- Can run with `python3 tests/test_pipeline.py` or `pytest tests/`

---

## Key Design Decisions

1. **Pure HTTP client** — no local model serving. All inference goes through the Swiss AI Stack's OpenAI-compatible API. This keeps the codebase simple and portable.

2. **Prompt-elicited reasoning only** — B3 uses three levels of prompt-elicited reasoning (no reasoning, free-form reasoning, structured criteria-based reasoning). All reasoning depth is controlled entirely by the prompt template — there is no native API thinking-budget feature involved. This keeps the design simple and portable across any OpenAI-compatible endpoint.

3. **Label extraction cascade** — the regex tries explicit verdict lines first (most reliable), then bare labels at end of text, then falls back to the last uppercase A/B/C in the text. `<think>` blocks are stripped before extraction to avoid matching reasoning content.

4. **Position bias uses same prompt_id** — both AB and BA conditions for the same pair share the original `prompt_id`. The `flipped()` method's `prompt_id + "_flipped"` is NOT used as the call's prompt_id. This lets `compute_position_bias()` match pairs by prompt_id.

5. **Semaphore-based concurrency** — `asyncio.Semaphore(8)` limits concurrent requests to avoid overwhelming the API. All calls in an experiment are dispatched via `asyncio.gather` for maximum parallelism within the limit.

6. **Exponential backoff** — on 429 or ClientError, waits `base_delay * 2^attempt` seconds (capped at 60s). Raises RuntimeError if all 5 retries exhausted.

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

# All experiments (200 pairs)
python3 run_experiments.py

# Single experiment
python3 run_experiments.py --experiments position_bias

# Quick smoke test
python3 run_experiments.py --n-pairs 20 --experiments position_bias

# Tests (no API key needed)
python3 tests/test_pipeline.py
```

---

## Project Status

- [x] Core pipeline implemented (inference client, templates, dataset loaders, experiments, metrics)
- [x] Unit tests (20/20 passing)
- [x] Architecture document (architecture.pdf)
- [x] README
- [x] Bug fixes and code review
- [ ] Run experiments on Swiss AI Stack (requires API access)
- [ ] Analysis and visualization of results
- [ ] Paper/report writing
