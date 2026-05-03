#!/usr/bin/env bash
# ============================================================================
# run_b3_baseline.sh
# ----------------------------------------------------------------------------
# Run three more B3 templates across the four difficulty buckets, to complete
# the comparison alongside reason_then_judge and structured_reasoning
# (already collected by run_b3_all_buckets.sh):
#
#   1. expert_rater                 — no-reasoning baseline
#   2. opro_llama                   — OPRO best prompt for Llama (no reasoning)
#   3. opro_llama_reason_then_judge — OPRO best prompt for Llama + reason
#                                     instruction
#
# These three together produce the full B1+B3 comparison set:
#   * expert_rater & opro_llama are technically B1 (template variants, no
#     reasoning) but are missing for the new 4-bucket split, so we collect
#     them here.
#   * opro_llama_reason_then_judge is the new combined B3 condition.
#
# Each bucket × template combination measures position bias (AB+BA) on
# 1500 pairs.
# Total: 4 buckets × 3 templates × 1500 × 2 = 36,000 judge calls.
#
# Outputs land in:
#     results/b3_by_difficulty/<bucket>/<template>_overall.jsonl
#
# Usage
# -----
#     ./run_b3_baseline.sh
# ============================================================================

set -euo pipefail

N_PAIRS=1500
TEMPLATES=(expert_rater opro_llama opro_llama_reason_then_judge)
BUCKETS=(easy medium hard tie)

for bucket in "${BUCKETS[@]}"; do
    for template in "${TEMPLATES[@]}"; do
        echo ""
        echo "============================================================"
        echo "  B3 — bucket: $bucket | template: $template"
        echo "============================================================"

        python run_b3_by_difficulty.py \
            --difficulty "$bucket" \
            --n-pairs "$N_PAIRS" \
            --run-name "$bucket" \
            --templates "$template"
    done
done

echo ""
echo "All baseline + OPRO + OPRO+reason runs complete. Results in:"
echo "  results/b3_by_difficulty/<bucket>/{expert_rater,opro_llama,opro_llama_reason_then_judge}_overall.jsonl"
