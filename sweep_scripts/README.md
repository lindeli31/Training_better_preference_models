# sweep_scripts

Scripts for running and plotting the B1 (position bias) sweep.

## Configuration — `config.py`

All sweep dimensions and plot style constants live in `config.py`.
Edit there to have the change propagate to the runner and all plot scripts:

| Setting | What it controls |
|---------|-----------------|
| `MODELS` | Full model IDs run in the sweep |
| `SWEEP_TEMPLATES` | Default templates (overridable via `--templates`) |
| `DIFFICULTIES` | Difficulty buckets to sweep over |
| `MODELS_ORDER` / `MODEL_LABELS` / `MODEL_COLORS` / `MODEL_HATCHES` | Plot style per model |
| `TEMPLATES_ORDER` / `TEMPLATE_LABELS` / `TEMPLATE_HATCHES` | Plot style per template |
| `BASELINE_TEMPLATES` / `OPTIMIZED_TEMPLATES` | Background shading in plots |

---

## B1 sweep — `run_b1_sweep.py`

Runs the position-bias experiment (B1) over all combinations of:

| Axis | Values |
|------|--------|
| **Models** | `apertus` (Apertus-70B), `llama33` (Llama-3.3-70B), `qwen35` (GLM-4.7-Flash) |
| **Templates** | `expert_rater`, `llm_judge`, `opro` |
| **Difficulty buckets** | `easy`, `medium`, `hard`, `tie` |

Each experiment runs on the full validation split for that bucket (no sub-sampling):

| Bucket | Validation pairs | Gold label |
|--------|-----------------|------------|
| easy   | 130             | A / B (~50/50) |
| medium | 112             | A / B (~50/50) |
| hard   | 171             | A / B (~50/50) |
| tie    | 106             | C (always)    |

Each pair is judged **twice** — once in the original A/B order and once flipped (B/A) — so one experiment makes `2 × n_pairs` API calls. The full default sweep is **36 experiments** (3 models × 3 templates × 4 buckets).

### Prerequisites

1. Set your API key:
   ```bash
   export SWISSAI_API_KEY=<your_key>
   ```
   Or add it to a `.env` file in the project root.

2. Ensure the processed dataset files exist under `data/`:
   ```
   data/helpsteer2_validation_full.json
   ```
   If missing, regenerate from the project root:
   ```bash
   python -m src.datasets.dataset
   ```

### Usage

```bash
# Run all 27 experiments (skips any that already have a metrics.json)
python sweep_scripts/run_b1_sweep.py

# Preview the plan without making any API calls
python sweep_scripts/run_b1_sweep.py --dry-run

# Re-run everything, overwriting existing results
python sweep_scripts/run_b1_sweep.py --force

# Restrict to specific models, templates, or difficulty buckets
python sweep_scripts/run_b1_sweep.py --models apertus llama33
python sweep_scripts/run_b1_sweep.py --templates expert_rater opro
python sweep_scripts/run_b1_sweep.py --difficulties easy hard tie

# Use optimised/model-specific templates (auto-selects variant, falls back to _llama)
python sweep_scripts/run_b1_sweep.py --templates gepa opro_tree

# Reduce concurrency if hitting rate limits (default: 8)
python sweep_scripts/run_b1_sweep.py --concurrency 4

# Write results to a custom directory
python sweep_scripts/run_b1_sweep.py --out results/my_run
```

### Output layout

```
results/b1_sweep/
  <model>/               # apertus / llama33 / qwen35
    <template>/          # expert_rater / llm_judge / opro
      <difficulty>/      # easy / medium / hard
        <template>_overall.jsonl   — raw judge-call records (one per call)
        metrics.json               — position-bias + accuracy metrics
```

`metrics.json` contains:
- `position_consistency` — fraction of pairs where flipping A/B does not change the verdict
- `position_bias_rate` — complement of position_consistency
- `bias_toward_first/second_position` — directional breakdown of inconsistencies
- `accuracy.ab_accuracy` — accuracy when the correct answer is in position A
- `accuracy.ba_accuracy` — accuracy when the correct answer is in position B
- `accuracy.accuracy_gap` — ab_accuracy − ba_accuracy (positive = first-position bias)
- `accuracy.overall_accuracy`

---

## Plotting — after the sweep

Run these from the **project root** once `results/b1_sweep/` is populated.

```bash
# Accuracy overview, position-bias overview, and summary heatmap
python sweep_scripts/plot_b1_sweep.py

# Stacked-bar accuracy plots broken down by difficulty / model / template
python sweep_scripts/plot_b1_accuracy.py

# Error-breakdown and position-consistency + accuracy summary
python sweep_scripts/plot_b1_errors.py
```

All figures are saved to `results/b1_sweep/figures/`. Pass `--out <dir>` to any
plot script to redirect output, or `--root <dir>` to point at a non-default
results directory.
