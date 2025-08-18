# prepare summarization data for training and evaluation
from datasets import load_dataset
import json
import os
def prepare_data_cnn_dailymail_sharedgpt(output_dir, prompt_postfix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = load_dataset("cnn_dailymail", "3.0.0")
    # prepare train data
    train_data = data["train"]
    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        for item in train_data:
            id = item["id"]
            user_value = item["article"]
            gpt_value = item["highlights"]
            data_item = {
                "id": id,
                'conversations': [
                    {'from': 'human', 'value': user_value.strip() + '\n\n' + prompt_postfix},
                    {'from': 'gpt', 'value': gpt_value.strip()}
                ]
            }
            f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
    # prepare validation data
    validation_data = data["validation"]
    with open(os.path.join(output_dir, "validation.jsonl"), "w") as f:
        for item in validation_data:
            id = item["id"]
            user_value = item["article"]
            gpt_value = item["highlights"]
            data_item = {
                "id": id,
                'conversations': [
                    {'from': 'human', 'value': user_value.strip() + '\n\n' + prompt_postfix},
                    {'from': 'gpt', 'value': gpt_value.strip()}
                ]
            }
            f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
    # prepare test data
    test_data = data["test"]
    with open(os.path.join(output_dir, "test.jsonl"), "w") as f:
        for item in test_data:
            id = item["id"]
            user_value = item["article"]
            gpt_value = item["highlights"]
            data_item = {
                "id": id,
                'conversations': [
                    {'from': 'human', 'value': user_value.strip() + '\n\n' + prompt_postfix},
                    {'from': 'gpt', 'value': gpt_value.strip()}
                ]
            }
            f.write(json.dumps(data_item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    output_dir = "data/formatted_sharegpt/cnn_dailymail"
    prompt_postfix = "Given the above text, please write a summary in the format of CNN article highlights. The summary is:"
    prepare_data_cnn_dailymail_sharedgpt(output_dir, prompt_postfix)