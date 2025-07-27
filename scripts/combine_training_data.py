#!/usr/bin/env python3
"""
Combine non-RCT and RCT training data into a single training dataset.

This script reads the training-split data from both formatted PubMed RCT and formatted non-RCT datasets
and combines them into a single training file for supervised fine-tuning.

The data from RCT and non-RCT are evenly distributed in a sequence of [RCT, non-RCT, RCT, non-RCT, ...]

Usage:
    python scripts/combine_training_data.py
"""

import os
import sys
import json
import argparse
from itertools import zip_longest


def is_valid_json(line):
    """Check if a line is valid JSON."""
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def main():
    """Main function to execute the data combination process."""
    parser = argparse.ArgumentParser(
        description="Combine RCT and non-RCT training-split data into a single dataset"
    )

    parser.add_argument(
        "--rct-train",
        type=str,
        default="data/formatted_sharegpt/rct/train.jsonl",
        help="Path to RCT training-split data (default: data/formatted_sharegpt/rct/train.jsonl)"
    )

    parser.add_argument(
        "--non-rct-train",
        type=str,
        default="data/formatted_sharegpt/non_rct/train.jsonl",
        help="Path to non-RCT training-split data (default: data/formatted_sharegpt/non_rct/train.jsonl)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/combined_train.jsonl",
        help="Path to output combined training file (default: data/combined_train.jsonl)"
    )

    args = parser.parse_args()

    # Check input files
    if not os.path.exists(args.rct_train):
        sys.exit(f"[ERROR] RCT training file not found: {args.rct_train}\n"
                 "[INFO] Please run the data processing pipeline first:\n"
                 "       ./run_data_processing.sh")

    if not os.path.exists(args.non_rct_train):
        sys.exit(f"[ERROR] Non-RCT training file not found: {args.non_rct_train}\n"
                 "[INFO] Please run the data processing pipeline first:\n"
                 "       ./run_data_processing.sh")

    merged_lines = 0
    skipped_rct = 0
    skipped_non_rct = 0

    with open(args.rct_train, "r", encoding="utf-8") as f_rct, \
         open(args.non_rct_train, "r", encoding="utf-8") as f_non_rct, \
         open(args.output, "w", encoding="utf-8") as f_output:

        for rct_line, non_rct_line in zip_longest(f_rct, f_non_rct):
            if rct_line:
                if is_valid_json(rct_line):
                    f_output.write(rct_line.rstrip('\n') + '\n')
                    merged_lines += 1
                else:
                    print(f"[WARNING] Skipping invalid RCT line: {rct_line.strip()}")
                    skipped_rct += 1

            if non_rct_line:
                if is_valid_json(non_rct_line):
                    f_output.write(non_rct_line.rstrip('\n') + '\n')
                    merged_lines += 1
                else:
                    print(f"[WARNING] Skipping invalid non-RCT line: {non_rct_line.strip()}")
                    skipped_non_rct += 1

    print(f"\n[INFO] Combined training data saved to: {args.output}")
    print(f"[INFO] Total merged lines: {merged_lines}")
    print(f"[INFO] Skipped RCT lines: {skipped_rct}")
    print(f"[INFO] Skipped non-RCT lines: {skipped_non_rct}")
    return 0


if __name__ == "__main__":
    sys.exit(main())