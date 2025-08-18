# make this file executable: chmod +x scripts/Process_acl_agd.py
# python scripts/Process_acl_agd.py ./data/processed/acl-agd train.jsonl
# python scripts/Process_acl_agd.py ./data/processed/acl-agd validation.jsonl
# python scripts/Process_acl_agd.py ./data/processed/acl-agd test.jsonl

import argparse
import os
import json

data_dir = "./data/raw/acl-agd"

def process_acl_agd(output_folder: str, file_name: str):
    """
    Process the ACL-AGD file and save it to the specified output folder as a JSONL file.
    Each line is a JSON object, suitable for large files.
    """
    file_path = os.path.join(data_dir, file_name)
    output_path = os.path.join(output_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"[INFO] Starting parsing: {file_path}")
    item_count = 0
    with open(file_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            data_item = json.loads(line)
            item = parse_item(data_item)
            if item:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                item_count += 1
    print(f"[INFO] Finished parsing. Total items: {item_count}")
    print(f"[INFO] Output written to: {output_path}")


def parse_item(data_item):
    """
    Parse a single ACL-AGD JSONL record into the unified schema:
    {
        "id": str,                  # from data_item["doc_key"]
        "title": "",               # title not provided in ACL-AGD
        "sentences": [              # aligned with data_item["sents_label"]
            {"label": str, "text": str},
            ...
        ]
    }

    The source uses tokenized text with punctuation as separate tokens.
    We split sentences by accumulating tokens until a sentence-ending token
    (one of {'.', '?', '!'}) is encountered.
    """
    if not isinstance(data_item, dict):
        return None

    doc_key = data_item.get("doc_key") or data_item.get("id")
    if not doc_key:
        return None

    raw_sents = data_item.get("sents")
    labels = data_item.get("sents_label") or []

    # If no labels are provided, skip the document
    if len(labels) == 0:
        return None

    # Normalize sentences into a list of strings
    sentences_text = []
    if isinstance(raw_sents, str):
        tokens = raw_sents.strip().split()
        current_tokens = []
        for tok in tokens:
            # if the token is a sentence-ending token, add the current sentence to the list
            if tok in {".", "?", "!"}:
                sentences_text.append(" ".join(current_tokens).strip() + tok)
                current_tokens = []
            else:
                # if the token is not a sentence-ending token, add it to the current sentence
                current_tokens.append(tok)
        if current_tokens:
            sentences_text.append(" ".join(current_tokens).strip())
    else:
        return None

    # Align lengths conservatively
    if len(labels) != len(sentences_text):
        return None

    sentences = []
    for i in range(len(labels)):
        label = str(labels[i]).strip().upper()
        text = sentences_text[i].strip()
        sentences.append({"label": label, "text": text})

    return {
        "id": doc_key,
        "title": "",
        "sentences": sentences,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Preprocess the ACL-AGD file."
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
    process_acl_agd(args.output_folder, args.file_name)