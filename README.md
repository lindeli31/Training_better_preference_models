# LLM Judge Bias & Sensitivity Experiments

Research pipeline for measuring systematic biases and sensitivities in LLM-as-a-judge evaluations on the [Swiss AI Stack](https://api.swissai.cscs.ch/).

## Project Structure

```
├── src/                        # Core library
│   ├── inference_client.py     # Async HTTP client (retry, backoff, label parsing)
│   ├── templates.py            # Judge prompt templates and evaluation criteria
│   ├── dataset.py              # HuggingFace dataset loaders → PairRecord objects
│   ├── experiments.py          # Experiment runners (B1–B4): build calls, dispatch, save JSONL
│   ├── metrics.py              # Metrics from JSONL results (bias rates, agreement, accuracy)
│   └── opro_position_bias.py   # OPRO prompt optimisation for reducing position bias
├── data/                       # Local dataset files (JSON + CSV)
│   ├── helpsteer2_train.json / .csv
│   ├── helpsteer2_train_full.json / .csv
│   ├── helpsteer2_validation.json / .csv
│   └── helpsteer2_validation_full.json / .csv
├── tests/
│   └── test_pipeline.py        # Unit tests (no API calls required)
├── docs/
│   ├── architecture.tex        # LaTeX source for architecture document
│   └── architecture.pdf        # Compiled architecture document
├── run_experiments.py          # CLI orchestrator — runs experiments and prints summaries
├── run_opro.py                 # OPRO prompt optimisation runner
├── check_models.py             # List available models on the Swiss AI Stack
├── dataset_analysis.ipynb      # Jupyter notebook for dataset exploration
├── requirements.txt            # Python dependencies
└── PROJECT.md                  # Detailed project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:
```bash
SWISSAI_API_KEY=<your-key>
SWISSAI_MODEL=meta-llama/Llama-3.3-70B-Instruct
```

`SWISSAI_MODEL` pins the model used by all scripts. To see what models are currently available:
```bash
python check_models.py
```

Then update `SWISSAI_MODEL` in `.env` accordingly. Both `run_experiments.py` and `run_opro.py` will validate that the pinned model is available at startup and exit with a clear error if not.

You can also override the model for a single run without changing `.env`:
```bash
python run_experiments.py --model <model_id>
python run_opro.py --model <model_id>
```

## Dataset

The pipeline uses [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2). Download and save locally (JSON + CSV):

```bash
python -m src.dataset
```

This creates two versions per split (train / validation):
- **Standard** (`helpsteer2_train.json`): `prompt_id, prompt, response_a, response_b, gold_label`
- **Full** (`helpsteer2_train_full.json`): adds individual scores (`helpfulness_a`, `correctness_a`, `coherence_a`, `complexity_a`, `verbosity_a`, same for `_b`), composite `score_a/b`, `score_gap`, `len_a/b`, and `difficulty` (easy/medium/hard based on score gap)

## Check Available Models

```bash
python check_models.py
```

Lists all models currently available on the Swiss AI Stack. Use the output to set `SWISSAI_MODEL` in your `.env` file. Note: available models may change between runs — pinning via `.env` ensures consistency, especially for multi-run methods like OPRO.

## Running Experiments

Get info about the evaluation script:
```bash
python -m src.scripts.evaluate_model --help
```

### All experiments (200 pairs, default settings)

```bash
python3 run_experiments.py
```

### Run a single experiment

```bash
python3 run_experiments.py --experiments position_bias
python3 run_experiments.py --experiments template_sensitivity
python3 run_experiments.py --experiments reasoning_depth
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
  --difficulty hard \
  --exclude-ties \
  --concurrency 8 \
  --output-dir results/run_001
```

`--difficulty` filters pairs by score gap before sampling (`easy` gap > 1.0, `medium` gap > 0.33, `hard` gap ≤ 0.33). Omit to use all pairs. Requires the full dataset variant (`helpsteer2_{split}_full.json`).

`--exclude-ties` omits pairs with `gold_label=C` from all accuracy computations. Useful when running on difficulty-filtered subsets: hard pairs in HelpSteer2 are all exact score ties (gold=C), so accuracy is always 0 without this flag.

### Run tests (no API key required)

```bash
python3 tests/test_pipeline.py
# or
python3 -m pytest tests/test_pipeline.py -v
```

## Experiments

Every question has two answer candidates: let's call them A and B.

And three answer choices are given to an LLM:
- "first answer candidate is better"
- "second answer candidate is better"
- "the answer candidates are equally good"
Note, that the exact formulations can be different -- we also explore this as part of our work. (TODO: confirm this statement that we try different formulations)

We also researched the case when the ties are excuded.

The core idea of this project is to explore how the order of the answer candidates can influence the LLM answer (the B1 section).
We also explore ways how we can mitigate the impact of the ordering on an LLM by:
- changing the LLM instructions (B2 section)
- allowing it to reason (B3 section)
We also look at how minor, semantically preserving changes to the question (TODO: specify, if it is changes to the LLM instructions, answer candidates, or question context) influence the LLM's performance.

We have two sets of metrics:
- The metrics that are used to describe the model's performance on the question level (local) across two permutation of the answer candidates (AB and BA)
- The metrics that are used to describe the model's performance across the questions (global)

NOTE:
- A, B -- slots for answer candidates

### B1 — influence of the answer candidates order on the LLM answer

**Local metrics**

| Metric        | Description                                                                                                              |
|---------------|--------------------------------------------------------------------------------------------------------------------------|
| `consistency` | Is 1 if a model give its preference to the same answer candidate, irrelently of the answer candidate order. 0 otherwise. |
| `accuracy`    | Is 1 if a model selects the correct answer under both answer candidate orders. 0 otherwise.                              |

**Global metrics**

| Metric                        | Description                                                                                                                                |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `consistency`                 | Fraction of questions where the model was consistent in selection across both answer candidate orders. 1.0 = no bias.                      | TODO: better formulation!
| `bias_toward_first_position`  | Fraction of questions where the model always picked answer candidate that goes first.                                                      |
| `bias_toward_second_position` | Fraction of questions where the model always picked answer candidate that goes second.                                                     |
| `tie_inconsistency_rate`      | Fraction of questions where a tie in one condition became a non-tie in the other.                                                          |
| `hard_accuracy`               | Fraction of questions where the model chosed correct answers under both permutations.                                                      |
| `soft_accuracy`               | Fraction of questions where the model chosed correct answers in at least one of the permutations.                                          |
| `ab_accuracy`                 | Fraction of questions where the model chosed correct answer when it was in the first position.                                             |
| `ba_accuracy`                 | Fraction of questions where the model chosed correct answer when it was in the second position.                                            |
| `accuracy_gap`                | AB accuracy − BA accuracy. A large positive gap means the judge is right in AB partly due to position preference, not quality recognition. |

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

### B3 — Reasoning Depth

Three conditions comparing prompt-elicited reasoning levels:

| Condition                | What the prompt tells the model |
|--------------------------|---------------------------------|
| `no_reasoning`           | Just output A, B, or C. No explanation. |
| `reason_then_judge`      | Explain your reasoning about strengths and weaknesses, then give your verdict. |
| `structured_reasoning`   | Rate each response on helpfulness, accuracy, and coherence, then give your verdict. |

All reasoning is prompt-elicited — the prompt itself asks the model to reason (or not). No native API thinking-budget feature is involved.

**Key metrics**: accuracy vs. gold label, agreement with `no_reasoning` baseline, average latency per condition.

**Result file**: `results/reasoning_depth/criterion_<criterion>.jsonl`

### B4 — Input / Wording Sensitivity

Minor paraphrasing of the `expert_rater` template (3 semantic-equivalent variants) crossed with 5 evaluation criteria (helpful, quality, accurate, harmless, concise).

**Key metrics**: pairwise agreement across wording variants; criterion-label drift.

**Result file**: `results/input_sensitivity/all.jsonl`

### OPRO — Prompt Optimisation for Position Bias

Uses OPRO (Optimization by PROmpting) to find the system prompt that minimises position bias. The LLM generates candidate prompts, each is evaluated on a training subset, and the best is validated on held-out data.

```bash
# Default: 10 iterations, 80 eval pairs, 150 validation pairs
python run_opro.py

# Quick test
python run_opro.py --n-iterations 3 --eval-pairs 20 --n-val 50
```

Results saved to `results/opro/opro_results.json`. The best prompt found is also registered as the `opro` template in `src/templates.py`:

```bash
python run_experiments.py --template opro --experiments position_bias
```

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

Rate limiting: exponential backoff with up to 5 retries, capped at 60s delay. Concurrency is controlled via an asyncio semaphore (default 8 concurrent requests).
