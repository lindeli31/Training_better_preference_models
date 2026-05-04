#!/usr/bin/env bash
# ============================================================================
# run_b4_chinese.sh
# ----------------------------------------------------------------------------
# Add the Chinese system-prompt variant to the B4 language sweep without
# re-running PL/DE/IT (already collected by run_b4_languages.sh).
#
# Each bucket measures position bias (AB+BA) on 1500 pairs.
# Total: 4 buckets * 1 template * 1500 * 2 = 12,000 judge calls.
#
# Outputs land in:
#     results/b4_by_difficulty/<bucket>/expert_rater_zh_overall.jsonl
#
# Usage
# -----
#     ./run_b4_chinese.sh
# ============================================================================

set -euo pipefail

N_PAIRS=1500
TEMPLATE=expert_rater_zh
BUCKETS=(easy medium hard tie)

for bucket in "${BUCKETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  B4 — bucket: $bucket | template: $TEMPLATE"
    echo "============================================================"

    python run_b3_by_difficulty.py \
        --difficulty "$bucket" \
        --n-pairs "$N_PAIRS" \
        --run-name "$bucket" \
        --templates "$TEMPLATE" \
        --experiment-name b4_by_difficulty
done

echo ""
echo "B4 Chinese runs complete. Results in:"
echo "  results/b4_by_difficulty/<bucket>/expert_rater_zh_overall.jsonl"
