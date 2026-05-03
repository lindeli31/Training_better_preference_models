#!/usr/bin/env bash
# ============================================================================
# run_b3_tree_of_thoughts.sh
# ----------------------------------------------------------------------------
# Run only the tree_of_thoughts_judge B3 reasoning template on the four
# difficulty buckets, to extend the B3 comparison set with the ToT prompt.
#
# Each bucket measures position bias (AB+BA) on 1500 pairs.
# Total: 4 buckets × 1 template × 1500 × 2 = 12,000 judge calls.
#
# Outputs land in:
#     results/b3_by_difficulty/<bucket>/tree_of_thoughts_judge_overall.jsonl
#
# Usage
# -----
#     ./run_b3_tree_of_thoughts.sh
# ============================================================================

set -euo pipefail

N_PAIRS=1500
TEMPLATE=tree_of_thoughts_judge
BUCKETS=(easy medium hard tie)

for bucket in "${BUCKETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  B3 ToT — bucket: $bucket"
    echo "============================================================"

    python run_b3_by_difficulty.py \
        --difficulty "$bucket" \
        --n-pairs "$N_PAIRS" \
        --run-name "$bucket" \
        --templates "$TEMPLATE"
done

echo ""
echo "Tree-of-Thoughts B3 runs complete. Results in:"
echo "  results/b3_by_difficulty/<bucket>/tree_of_thoughts_judge_overall.jsonl"
