# python scripts/Process_PubMed_NonRCT.py ./data/processed/non_rct examples.txt
# python scripts/Process_PubMed_NonRCT.py ./data/processed/non_rct train_clean.txt
# python scripts/Process_PubMed_NonRCT.py ./data/processed/non_rct dev_clean.txt
# python scripts/Process_PubMed_NonRCT.py ./data/processed/non_rct test_clean.txt

import argparse
import os
import re
import json

data_dir = "./data/raw/abstractAnalysis/PubMedData/output"

def process_pubmed_non_rct(output_folder: str, file_name: str):
    """
    Process the PubMed Non-RCT file and save it to the specified output folder as a JSONL file.
    Each line is a JSON object, suitable for large files.
    """
    file_path = os.path.join(data_dir, file_name)
    # remove the .txt extension
    file_name = file_name.split(".")[0]
    output_path = os.path.join(output_folder, file_name + ".jsonl")
    os.makedirs(output_folder, exist_ok=True)

    print(f"[INFO] Starting parsing: {file_path}")
    item_count = 0
    with open(file_path, "r") as f_in, open(output_path, "w") as f_out:
        lines = []
        for line in f_in:
            if line.startswith('### '):
                if lines:
                    item = parse_item(lines)
                    if item:
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        item_count += 1
                        if item_count % 50 == 0:
                            print(f"[INFO] Parsed {item_count} items...")
                lines = [line]
            else:
                # if the line is not empty, add it to the current item
                if line.strip():
                    lines.append(line)
        # Save the last item
        if lines:
            item = parse_item(lines)
            if item:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                item_count += 1
    print(f"[INFO] Finished parsing. Total items: {item_count}")
    print(f"[INFO] Output written to: {output_path}")



def parse_item(lines):
    """
    Parse a single data item given an iterator of lines.
    Returns a dict in the following format:
    {
        "id": "1558345",
        "title": "The idle intravenous catheter.",
        "sentences": [
            {
                "label": "OBJECTIVE",
                "text": "......."
            },
            {
                "label": "METHODS",
                "text": "......."
            },
            ...
        ]
    }
    Stops at the first blank line after a section.
    """
    item = {}
    sentences = []
    # Read ID
    lines_iter = iter(lines)
    for line in lines_iter:
        if line.startswith('### '):
            item['id'] = line[4:].strip()
            break
    else:
        return None  # No more items
    # Read Title
    for line in lines_iter:
        if line.startswith('#### '):
            item['title'] = line[5:].strip()
            break
    else:
        return None
    # Read sentences
    for line in lines_iter:
        line = line.rstrip()
        if not line:
            continue  # skip blank lines
        m = re.match(r'^([A-Z]+)\s+(.+)$', line)
        if m:
            label, text = m.groups()
            sentences.append({"label": label, "text": text.strip()})
        elif sentences:
            # Continuation of previous sentence
            sentences[-1]["text"] += ' ' + line.strip()
    item['sentences'] = sentences
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Preprocess the PubMed Non-RCT file."
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="The path to the folder where the dataset should be saved.",
    )
    parser.add_argument(
        "file_name",
        type=str,
        help="The name of the input file to process.",
    )
    args = parser.parse_args()
    process_pubmed_non_rct(args.output_folder, args.file_name)
