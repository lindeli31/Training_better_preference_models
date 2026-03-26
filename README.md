# LLM Judge Bias & Sensitivity Experiments

Research pipeline for measuring systematic biases and sensitivities in LLM-as-a-judge evaluations using **Qwen3-30B-A3B-Instruct-2507** on the [Swiss AI Stack](https://serving.swissai.svc.cscs.ch/).

## Project Structure

```
inference_client.py   # Async HTTP client for the Swiss AI Stack (retry, backoff, label parsing)
templates.py          # All judge prompt templates and evaluation criteria
dataset.py            # HuggingFace dataset loaders → normalised PairRecord objects
experiments.py        # Experiment runners (B1–B4): build calls, dispatch, save JSONL
metrics.py            # Compute metrics from JSONL results (bias rates, agreement, accuracy)
run_experiments.py    # CLI orchestrator — runs experiments and prints summaries
test_pipeline.py      # Unit tests (no API calls required)
requirements.txt      # Python dependencies
architecture.pdf      # Detailed pipeline architecture document
```

## Setup

```bash
pip install -r requirements.txt
export SWISSAI_API_KEY="<your-key>"
```

## Running Experiments

### All experiments (200 pairs, default settings)

```bash
python3 run_experiments.py
```

### Run a single experiment

```bash
python3 run_experiments.py --experiments position_bias
python3 run_experiments.py --experiments template_sensitivity
python3 run_experiments.py --experiments thinking_budget
python3 run_experiments.py --experiments input_sensitivity
```

### Quick smoke test (20 pairs)

```bash
python3 run_experiments.py --n-pairs 20 --experiments position_bias
```

### Full configuration

```bash
python3 run_experiments.py \
  --dataset nvidia/HelpSteer2 \
  --split validation \
  --n-pairs 200 \
  --criterion helpful \
  --thinking-budget 1024 \
  --concurrency 8 \
  --output-dir results/run_001
```

### Run tests (no API key required)

```bash
python3 test_pipeline.py
# or
python3 -m pytest test_pipeline.py -v
```

## Experiments

### B1 — Position Bias

Every pair is judged twice: once as (A=chosen, B=rejected) and once flipped (A=rejected, B=chosen). A consistent judge should flip its label accordingly.

**Key metric**: `position_consistency` — fraction of pairs where label is stable across both orderings. `1.0` = no bias.

**Result file**: `results/position_bias/<template>_<criterion>.jsonl`

### B2 — Template Sensitivity

The same pairs are judged using 5 system-prompt variants (same data, same order):

| Template ID    | Description                          |
|----------------|--------------------------------------|
| `expert_rater` | "You are an expert rater..." (baseline) |
| `llm_judge`    | "You are an LLM used to judge..."    |
| `neutral`      | Neutral imperative framing           |
| `academic`     | Academic/formal evaluator framing    |
| `minimal`      | One-liner: "Pick the better response" |

**Key metric**: `overall_pairwise_agreement` across templates.

**Result file**: `results/template_sensitivity/criterion_<criterion>.jsonl`

### B3 — Prompt-Elicited Reasoning

Five conditions comparing reasoning depth:

| Condition      | Mechanism                                       | Thinking tokens |
|----------------|-------------------------------------------------|-----------------|
| `no_thinking`  | Base template, no CoT                           | 0               |
| `cot_prompted` | Template instructs model to reason step-by-step | 0               |
| `think_512`    | Qwen3 native thinking mode                      | 512             |
| `think_1024`   | Qwen3 native thinking mode                      | 1024            |
| `think_2048`   | Qwen3 native thinking mode                      | 2048            |

> **Note**: `cot_prompted` tests prompt-elicited Chain of Thought (the model is explicitly told to argue its decision before answering). This is not a native API thinking-budget feature — it is a prompt-level instruction. The `think_*` conditions use Qwen3's optional thinking mode via API parameters.

**Key metrics**: accuracy vs. gold label, agreement with `no_thinking` baseline, average latency per condition.

**Result file**: `results/thinking_budget/criterion_<criterion>.jsonl`

### B4 — Input / Wording Sensitivity

Minor paraphrasing of the `expert_rater` template (3 semantic-equivalent variants) crossed with 5 evaluation criteria (helpful, quality, accurate, harmless, concise).

**Key metrics**: pairwise agreement across wording variants; criterion-label drift.

**Result file**: `results/input_sensitivity/all.jsonl`

## Output Format

Every JSONL line is a `JudgeResponse`:

```json
{
  "prompt_id":     "a3f9b12c10",
  "experiment_id": "position_bias",
  "condition":     "AB",
  "raw_text":      "Response A is more helpful...\nVerdict: A",
  "label":         "A",
  "thinking":      null,
  "latency_s":     1.234,
  "total_tokens":  512,
  "parse_ok":      true,
  "error":         null
}
```

| Field          | Description                                              |
|----------------|----------------------------------------------------------|
| `prompt_id`    | Unique identifier for the dataset pair                   |
| `experiment_id`| Which experiment produced this result                    |
| `condition`    | Experimental condition (e.g. `"AB"`, `"BA"`, template)   |
| `raw_text`     | Full model output                                        |
| `label`        | Extracted verdict: `"A"`, `"B"`, `"C"` (tie), or `null` |
| `thinking`     | Extracted `<think>...</think>` block if present          |
| `latency_s`    | Request latency in seconds                               |
| `total_tokens` | Total tokens used (prompt + completion)                  |
| `parse_ok`     | Whether label extraction succeeded                       |
| `error`        | Error message if the request failed                      |

## API Details

The client talks to the OpenAI-compatible endpoint at the Swiss AI Stack:

```
POST https://serving.swissai.svc.cscs.ch/v1/chat/completions
```

Qwen3 thinking mode is activated via extra body parameters:

```json
{
  "chat_template_kwargs": {"enable_thinking": true},
  "thinking": {"type": "enabled", "budget_tokens": 1024}
}
```

Rate limiting: exponential backoff with up to 5 retries, capped at 60s delay. Concurrency is controlled via an asyncio semaphore (default 8 concurrent requests).
