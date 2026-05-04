#!/usr/bin/env bash
# ============================================================================
# run_b4_languages.sh
# ----------------------------------------------------------------------------
# Experiment B4 — input sensitivity by system-prompt language.
# Position bias is measured (AB+BA) on each of the four difficulty buckets
# using the expert_rater system prompt translated into Polish, German and
# Italian (output instructions also translated; user prompt stays in English).
#
# The English baseline (expert_rater) is NOT re-run here — it already exists
# under results/b3_by_difficulty/<bucket>/expert_rater_overall.jsonl from
# run_b3_baseline.sh. Plotting code that consumes B4 should pick the EN
# baseline up from there.
#
# Each bucket × language combination measures position bias (AB+BA) on
# 1500 pairs.
# Total: 4 buckets × 3 languages × 1500 × 2 = 36,000 judge calls.
#
# Outputs land in:
#     results/b4_by_difficulty/<bucket>/<template>_overall.jsonl
#
# Usage
# -----
#     ./run_b4_languages.sh
# ============================================================================

set -euo pipefail

N_PAIRS=1500
TEMPLATES=(expert_rater_pl expert_rater_de expert_rater_it expert_rater_zh)
BUCKETS=(easy medium hard tie)

for bucket in "${BUCKETS[@]}"; do
    for template in "${TEMPLATES[@]}"; do
        echo ""
        echo "============================================================"
        echo "  B4 — bucket: $bucket | template: $template"
        echo "============================================================"

        python run_b3_by_difficulty.py \
            --difficulty "$bucket" \
            --n-pairs "$N_PAIRS" \
            --run-name "$bucket" \
            --templates "$template" \
            --experiment-name b4_by_difficulty
    done
done

echo ""
echo "B4 language runs complete. Results in:"
echo "  results/b4_by_difficulty/<bucket>/expert_rater_{pl,de,it,zh}_overall.jsonl"
echo ""
echo "EN baseline already at:"
echo "  results/b3_by_difficulty/<bucket>/expert_rater_overall.jsonl"
