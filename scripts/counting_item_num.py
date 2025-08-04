# count the number of items
# python scripts/counting_item_num.py --input_path data/formatted_sharegpt/rct/train.jsonl
# python scripts/counting_item_num.py --input_path data/formatted_sharegpt/rct/test.jsonl
# python scripts/counting_item_num.py --input_path data/formatted_sharegpt/rct/dev.jsonl
# python scripts/counting_item_num.py --input_path data/formatted_sharegpt/non_rct/train.jsonl
# python scripts/counting_item_num.py --input_path data/formatted_sharegpt/non_rct/test.jsonl
# python scripts/counting_item_num.py --input_path data/formatted_sharegpt/non_rct/dev.jsonl

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()

    item_num = 0

    with open(args.input_path, "r") as f:
        for line in f:
            item_num += 1

    print(f"The number of items in {args.input_path} is {item_num}")