# python scripts/formatting.py data/filtered/rct/dev.jsonl data/formatted_sharegpt/rct/dev.jsonl --format sharegpt --use_prompt_index 0
# python scripts/formatting.py data/filtered/rct/test.jsonl data/formatted_sharegpt/rct/test.jsonl --format sharegpt --use_prompt_index 0
# python scripts/formatting.py data/filtered/rct/train.jsonl data/formatted_sharegpt/rct/train.jsonl --format sharegpt --use_prompt_index 0

# python scripts/formatting.py data/filtered/non_rct/dev_clean.jsonl data/formatted_sharegpt/non_rct/dev.jsonl --format sharegpt --use_prompt_index 0
# python scripts/formatting.py data/filtered/non_rct/test_clean.jsonl data/formatted_sharegpt/non_rct/test.jsonl --format sharegpt --use_prompt_index 0
# python scripts/formatting.py data/filtered/non_rct/train_clean.jsonl data/formatted_sharegpt/non_rct/train.jsonl --format sharegpt --use_prompt_index 0
# python scripts/formatting.py data/filtered/non_rct/examples.jsonl data/formatted_sharegpt/non_rct/examples.jsonl --format sharegpt --use_prompt_index 0

import json
import argparse
import os

PROMPTs = [
    "Given the above text, please write a conclusion section. The conclusion section is:",
    "Given the above text, please write a conclusion section in the format of PubMed paper abstract. The conclusion section is:",
    "Given the above text, please write a {} sentences conclusion section. The conclusion section is:",
    "Given the above text, please write a {} sentences conclusion section in the format of PubMed paper abstract. The conclusion section is:",
]
###################################################################
# Formatting the data into Alpaca and Sharegpt format
# The format of the input data must be a jsonl file with each line is a json object:
# {
#     "id": "1",
#     "title": "The idle intravenous catheter.",
#     "sentences": [
#         {"label": "OBJECTIVE", "text": "......."},
#         {"label": "METHODS", "text": "......."},
#         {"label": "RESULTS", "text": "......."},
#         {"label": "CONCLUSIONS", "text": "......."}
#     ]
# },
# ...
# {
#     "id": "10001",
#     "title": "The idle intravenous catheter.",
#     "sentences": [
#         {"label": "OBJECTIVE", "text": "......."},
#         {"label": "METHODS", "text": "......."},
#         {"label": "RESULTS", "text": "......."},
#         {"label": "CONCLUSIONS", "text": "......."}
#     ]
# }
# Alpaca and sharegpt format: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
###################################################################

def format_alpaca(data_path: str, output_path: str, use_prompt_index: int):
    """
    Format the data into Alpaca format
    """
    with open(data_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            sentences = item.get('sentences', [])
            # Compose instruction and input
            input_text = ""
            output_text = ""
            # counting the number of sentences in the conclusion
            num_conclusion_sentences = 0
            for sentence in sentences:
                if sentence['label'] != 'CONCLUSIONS':
                    input_text += sentence['text'] + ' '
                else:
                    # Count the number of sentences in the conclusion
                    num_conclusion_sentences += 1
                    output_text += sentence['text'] + ' '
            # Compose instruction
            instruction = PROMPTs[use_prompt_index].format(num_conclusion_sentences)
            if len(input_text) > 0 and len(output_text) > 0:
                output_item = {
                    'id': item['id'],
                    'title': item['title'],
                    'sentences': item['sentences'],
                    'alpaca_format': {
                                'instruction': instruction,
                                'input': input_text.strip(),
                                'output': output_text.strip()
                            }
                }
                fout.write(json.dumps(output_item) + '\n')

def format_sharegpt(data_path: str, output_path: str, use_prompt_index: int):
    """
    Format the data into Sharegpt format
    """
    with open(data_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            sentences = item.get('sentences', [])
            user_value = ""
            gpt_value = ""
            # counting the number of sentences in the conclusion
            num_conclusion_sentences = 0
            for sentence in sentences:
                if sentence['label'] != 'CONCLUSIONS':
                    user_value += sentence['text'] + ' '
                else:
                    # Count the number of sentences in the conclusion
                    num_conclusion_sentences += 1
                    gpt_value += sentence['text'] + ' '
            if len(user_value) > 0 and len(gpt_value) > 0:
                prompt_postfix = PROMPTs[use_prompt_index].format(num_conclusion_sentences)
                output_item = {
                    'id': item['id'],
                    'title': item['title'],
                    'sentences': item['sentences'],
                    'conversations': [
                        {'from': 'human', 'value': user_value.strip() + '\n\n' + prompt_postfix},
                        {'from': 'gpt', 'value': gpt_value.strip()}
                    ]
                }
                fout.write(json.dumps(output_item) + '\n')

if __name__ == '__main__': 
    # Read input file from command line argument
    parser = argparse.ArgumentParser(
        description="Format dataset into Alpaca or ShareGPT format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file to process"
    )
    parser.add_argument(
        "output_file", 
        type=str,
        help="Path to the file where formatted data should be saved"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["alpaca", "sharegpt"],
        default="alpaca",
        help="Output format - either 'alpaca' or 'sharegpt' (default: alpaca)"
    )
    parser.add_argument(
        "--use_prompt_index",
        type=int,
        required=True,
        help="Use the prompt with the given index, now there are 4 prompts: 0, 1, 2, 3"
    )

    args = parser.parse_args()

    if args.use_prompt_index not in [0, 1, 2, 3]:
        raise ValueError(f"Invalid prompt index: {args.use_prompt_index}, must be 0, 1, 2, 3")
    else:
        print(f"Using prompt: {PROMPTs[args.use_prompt_index]}")


    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Format data based on specified format
    if args.format == "alpaca":
        format_alpaca(args.input_file, args.output_file, args.use_prompt_index)
    elif args.format == "sharegpt":
        format_sharegpt(args.input_file, args.output_file, args.use_prompt_index)
    else:
        raise ValueError(f"Invalid format: {args.format}")

    print(f"Successfully formatted items into {args.format} format")
    print(f"Output written to: {args.output_file}")
