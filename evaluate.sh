#!/usr/bin/env bash

# MODEL="llama3.1-8b-base"
# MODEL="llama3.1-8b-rct10k"
# MODEL="llama3.1-8b-rct50k"
# MODEL="llama3.2-3b-base"
# MODEL="llama3.2-3b-rct10k"
  MODEL="llama3.2-3b-rct50k"

PRED_DIR="output/predictions/${MODEL}"
EVAL_DIR="output/eval_results/${MODEL}"

for i in 0 1 2 3; do
  non_rct_file="${PRED_DIR}/prompt${i}/non_rct.jsonl"
  rct_file="${PRED_DIR}/prompt${i}/rct.jsonl"

  echo "==== Start evaluating the non-RCT results of prompt $i ===="
  if [[ -f "$non_rct_file" ]]; then
    python scripts/evaluate_result.py \
      --input_path "$non_rct_file" \
      --output_path "${EVAL_DIR}/prompt${i}/non_rct_results.jsonl"
  else
    echo "File not found: $non_rct_file — skipping."
  fi

  echo "==== Start evaluating the RCT results of prompt $i ===="
  if [[ -f "$rct_file" ]]; then
    python scripts/evaluate_result.py \
      --input_path "$rct_file" \
      --output_path "${EVAL_DIR}/prompt${i}/rct_results.jsonl"
  else
    echo "File not found: $rct_file — skipping."
  fi
done