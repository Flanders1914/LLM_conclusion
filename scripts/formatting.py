# python scripts/formatting.py data/rct_filtered/dev_filtered.jsonl data/rct_formatted --format sharegpt
# python scripts/formatting.py data/non_rct_filtered/dev_clean_filtered.jsonl data/non_rct_formatted --format sharegpt
# python scripts/formatting.py data/non_rct_filtered/examples_filtered.jsonl data/non_rct_formatted --format sharegpt
import json
import argparse
import os
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

def format_alpaca(data_path: str, output_path: str):
    """
    Format the data into Alpaca format
    """
    with open(data_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            sentences = item.get('sentences', [])
            # Compose instruction and input
            instruction = "Make a scientific conclusion based on the given text:"
            input_text = ""
            output_text = ""
            for sentence in sentences:
                if sentence['label'] != 'CONCLUSIONS':
                    input_text += sentence['text'] + ' '
                else:
                    output_text += sentence['text'] + ' '
            if len(input_text) > 0 and len(output_text) > 0:
                output_item = {
                    'instruction': instruction,
                    'input': input_text.strip(),
                    'output': output_text.strip()
                }
                fout.write(json.dumps(output_item) + '\n')

def format_sharegpt(data_path: str, output_path: str):
    """
    Format the data into Sharegpt format
    """
    with open(data_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            sentences = item.get('sentences', [])
            user_value = ""
            gpt_value = ""
            for sentence in sentences:
                if sentence['label'] != 'CONCLUSIONS':
                    user_value += sentence['text'] + ' '
                else:
                    gpt_value += sentence['text'] + ' '
            if len(user_value) > 0 and len(gpt_value) > 0:
                output_item = {
                    'conversations': [
                        {'from': 'human', 'value': user_value.strip()},
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
        "output_folder", 
        type=str,
        help="Path to the folder where formatted data should be saved"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["alpaca", "sharegpt"],
        default="alpaca",
        help="Output format - either 'alpaca' or 'sharegpt' (default: alpaca)"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    # extract the file name
    file_name = os.path.basename(args.input_file)
    file_name = file_name.split(".")[0]

    # Format data based on specified format
    if args.format == "alpaca":
        output_file = os.path.join(args.output_folder, f"{file_name}_alpaca.jsonl")
        format_alpaca(args.input_file, output_file)
    elif args.format == "sharegpt":
        output_file = os.path.join(args.output_folder, f"{file_name}_sharegpt.jsonl")
        format_sharegpt(args.input_file, output_file)
    else:
        raise ValueError(f"Invalid format: {args.format}")

    print(f"Successfully formatted items into {args.format} format")
    print(f"Output written to: {output_file}")
