#!/bin/bash

buckets=("impossible" "gap_0.33-0.67" "gap_0.67-1.00" "gap_1.00-1.33" "gap_1.33-1.67" "gap_1.67-2.00" "gap_2.00+")

for bucket in "${buckets[@]}"; do
  echo "Running $bucket..."
  python run_evaluate_accuracy.py --difficulty "$bucket" --n-pairs 1500 --run-name "bucket_$bucket" 2>&1 | grep -A 25 "Accuracy breakdown" | head -20
  echo "---"
done
