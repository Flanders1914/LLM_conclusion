import os
import json
import argparse

from datasets import load_dataset

###################################################################
# Filter the dataset
# Supported datasets:
# - pubmed_rct
# - python scripts/filter.py data/processed/rct/dev.jsonl data/filtered/rct
# - python scripts/filter.py data/processed/rct/test.jsonl data/filtered/rct
# - python scripts/filter.py data/processed/rct/train.jsonl data/filtered/rct

# - pubmed_non_rct
# - python scripts/filter.py data/processed/non_rct/dev_clean.jsonl data/filtered/non_rct
# - python scripts/filter.py data/processed/non_rct/test_clean.jsonl data/filtered/non_rct
# - python scripts/filter.py data/processed/non_rct/train_clean.jsonl data/filtered/non_rct
# - python scripts/filter.py data/processed/non_rct/examples.jsonl data/filtered/non_rct

# - acl_agd
# - python scripts/filter.py data/processed/acl-agd/train.jsonl data/filtered/acl-agd
# - python scripts/filter.py data/processed/acl-agd/validation.jsonl data/filtered/acl-agd
# - python scripts/filter.py data/processed/acl-agd/test.jsonl data/filtered/acl-agd

###################################################################

def loading_data(path: str):
    """
    Load the pubmed_non_rct dataset
    """
    ds = load_dataset("json", data_files=path)
    print(f"success load data from {path}")
    # The data is in the default split "train"
    print("Here is the snapshot of the dataset:")
    print(ds['train'])
    print(ds['train'][0])
    print(ds['train'][0]['sentences'][0])
    print("--------------------------------")
    return ds['train']

def filter_items(ds):
    """
    Filter the pubmed_rct dataset
    """
    filtered_items = []
    count = 0
    for item in ds:
        if count % 50 == 0:
            print(f"Processing the {count}th item")
        count += 1
        sentences = item.get('sentences', [])
        labels = [s['label'] for s in sentences]
        # 1. Must have METHODS and RESULTS labelled sentences
        if 'METHODS' not in labels or 'RESULTS' not in labels:
            continue
        # 2. Must have at least 1 CONCLUSIONS labelled sentence
        n_conclusions = labels.count('CONCLUSIONS')
        if n_conclusions < 1:
            continue
        # 3. CONCLUSIONS labelled sentences must be the last sentences of the item
        if n_conclusions > 0:
            if labels[-n_conclusions:] != ['CONCLUSIONS'] * n_conclusions:
                continue
        # 4. The number of CONCLUSIONS labelled sentences must be smaller than the number of METHODS and RESULTS labelled sentences
        n_methods = labels.count('METHODS')
        n_results = labels.count('RESULTS')
        if n_conclusions >= (n_methods + n_results):
            continue
        filtered_items.append(item)
    return filtered_items

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Filter the dataset"
    )

    parser.add_argument(
        "Input_file_path",
        type=str,
        help="The path to the input file",
    )
    parser.add_argument(
        "Output_folder",
        type=str,
        help="The path to the folder where the dataset should be saved.",
    )

    args = parser.parse_args()
    file_name = args.Input_file_path.split("/")[-1]
    # remove the file extension
    file_name = file_name.split(".")[0]
    print(f"Start filtering the dataset in {file_name} from {args.Input_file_path}")
    print("--------------------------------")
    ds = loading_data(args.Input_file_path)
    filtered_items = filter_items(ds)
    print(f"The number of filtered items is {len(filtered_items)}")
    if len(filtered_items) > 0:
        print(f"The first filtered item is {filtered_items[0]}")
        print(f"The last filtered item is {filtered_items[-1]}")
    else:
        print("No items passed the filtering criteria.")
    print("--------------------------------")
    print("Save the filtered items to the output file")
    
    os.makedirs(args.Output_folder, exist_ok=True)
    output_file_path = os.path.join(args.Output_folder, f"{file_name}.jsonl")
    with open(output_file_path, 'w') as f:
        for item in filtered_items:
            f.write(json.dumps(item) + '\n')
    print(f"Successfully saved the filtered items to {output_file_path}")