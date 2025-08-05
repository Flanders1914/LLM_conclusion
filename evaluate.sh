#!/usr/bin/env bash

BASE_DIR="output/predictions/llama3_8b_base"
TOP_N=10

for i in 0 1 2 3; do
  echo "==== Prompt $i: non_rct ===="
  python scripts/evaluate_result.py \
    --input_path "${BASE_DIR}/non_rct_prompt${i}.jsonl" \
    --output_path "${BASE_DIR}/non_rct_prompt${i}_result.json" \
    --top_n ${TOP_N} \
    >> "${BASE_DIR}/non_rct_prompt${i}_result.txt"
  
  echo "==== Prompt $i: rct ===="
  python scripts/evaluate_result.py \
    --input_path "${BASE_DIR}/rct_prompt${i}.jsonl" \
    --output_path "${BASE_DIR}/rct_prompt${i}_result.json" \
    --top_n ${TOP_N} \
    >> "${BASE_DIR}/rct_prompt${i}_result.txt"
done