#!/usr/bin/env bash
# ============================================================================
# run_b3_opro_aligned.sh
# ----------------------------------------------------------------------------
# Re-run B3 for the two OPRO templates after aligning their definitions with
# main's SYSTEM_OPRO_LLAMA (the verbatim OPRO best prompt, with the OPRO
# output instruction embedded inline rather than appended).
#
#   1. opro_llama                   — SYSTEM_OPRO_LLAMA used as-is
#                                     (single-letter output instruction is the
#                                     last paragraph of the OPRO body itself).
#   2. opro_llama_reason_then_judge — SYSTEM_OPRO_LLAMA followed by an explicit
#                                     "reason step-by-step over the four
#                                     criteria, then output your verdict"
#                                     instruction. The OPRO output rule still
#                                     applies; the model is just asked to
#                                     think before applying it.
#
# Output overwrites the earlier (mis-defined) opro_llama runs in
#     results/b3_by_difficulty/<bucket>/{opro_llama,opro_llama_reason_then_judge}_overall.jsonl
#
# Each bucket × template = 1500 pairs × (AB+BA) = 3000 calls.
# Total: 4 buckets × 2 templates × 3000 = 24,000 judge calls.
#
# Usage
# -----
#     ./run_b3_opro_aligned.sh
# ============================================================================

set -euo pipefail

N_PAIRS=1500
TEMPLATES=(opro_llama opro_llama_reason_then_judge)
BUCKETS=(easy medium hard tie)

for bucket in "${BUCKETS[@]}"; do
    for template in "${TEMPLATES[@]}"; do
        echo ""
        echo "============================================================"
        echo "  B3 (aligned) — bucket: $bucket | template: $template"
        echo "============================================================"

        python run_b3_by_difficulty.py \
            --difficulty "$bucket" \
            --n-pairs "$N_PAIRS" \
            --run-name "$bucket" \
            --templates "$template"
    done
done

echo ""
echo "Aligned OPRO B3 runs complete. Results in:"
echo "  results/b3_by_difficulty/<bucket>/opro_llama_overall.jsonl"
echo "  results/b3_by_difficulty/<bucket>/opro_llama_reason_then_judge_overall.jsonl"
