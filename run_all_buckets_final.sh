#!/bin/bash

buckets=("tie" "hard" "medium" "easy")

for bucket in "${buckets[@]}"; do
  echo "=== $bucket ==="
  python run_evaluate_accuracy.py --difficulty "$bucket" --n-pairs 1500 --run-name "final_$bucket" 2>&1 | grep -A 15 "Accuracy breakdown" | head -16
done
