#!/usr/bin/env bash
# ============================================================================
# run_b3_all_buckets.sh
# ----------------------------------------------------------------------------
# Run the B3 reasoning-depth experiment across the four difficulty buckets,
# using only the two reasoning templates (skipping the no-reasoning baseline,
# which was already covered by the accuracy evaluation runs).
#
# Each bucket × template combination measures position bias (AB + BA) on
# 1500 pairs.  Total: 4 buckets × 2 templates × 1500 × 2 = 24,000 judge calls.
#
# Outputs land in:
#     results/b3_by_difficulty/<bucket>/<template>_overall.jsonl
#
# Usage
# -----
#     ./run_b3_all_buckets.sh
#
# Note: load_dataset_pairs already filters by `difficulty=tie`, and every
# tie pair has gold_label=C by construction, so no --pct-ties flag is needed.
# ============================================================================

set -euo pipefail

N_PAIRS=1500
TEMPLATES=(reason_then_judge structured_reasoning)
BUCKETS=(easy medium hard tie)

for bucket in "${BUCKETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  B3 — bucket: $bucket"
    echo "============================================================"

    python run_b3_by_difficulty.py \
        --difficulty "$bucket" \
        --n-pairs "$N_PAIRS" \
        --run-name "$bucket" \
        --templates "${TEMPLATES[@]}"
done

echo ""
echo "All B3 runs complete. Results in: results/b3_by_difficulty/<bucket>/"
