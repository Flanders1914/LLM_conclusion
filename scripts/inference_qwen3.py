# CUDA_VISIBLE_DEVICES=0 python scripts/inference_qwen3.py --model_name qwen3 --data_path data/formatted_sharegpt/non_rct/examples.jsonl --output_path output/qwen3/non_rct_examples.jsonl
# CUDA_VISIBLE_DEVICES=0 python scripts/inference_qwen3.py --model_name qwen3 --data_path data/formatted_sharegpt/rct/dev.jsonl --output_path output/qwen3/rct_dev.jsonl
# CUDA_VISIBLE_DEVICES=0 python scripts/inference_qwen3.py --model_name qwen3 --data_path data/formatted_sharegpt/non_rct/dev.jsonl --output_path output/qwen3/non_rct_dev.jsonl
# Only support sharegpt style for now

from unsloth import FastLanguageModel
import argparse
import json
import os
import re

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
TEST_NUM = 1000

fourbit_models = [
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", # Qwen 14B 2x faster
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for the model, only support sharegpt style for now"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model to use"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="The path to the data to use"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The path to the output to use"
    )
    args = parser.parse_args()

    if args.model_name == "qwen3":
        model_name = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    # create the output directory if it doesn't exist
    dirpath = os.path.dirname(args.output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-14B",
        max_seq_length = 2048,   # Context length - can be longer, but uses more memory
        load_in_4bit = True,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
    )

    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    # Start inference
    count = 0
    print(f"Start inference, model: {model_name}, data: {args.data_path}, output: {args.output_path}")
    print("------------------------------------------------------------------------------------------------")

    with open(args.data_path, 'r') as fin, open(args.output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            input_text = item['conversations'][0]['value']
            reference_text = item['conversations'][1]['value']
            messages = [
                {"role": "user",
                "content": input_text},
            ]
            # prepare the input ids
            text = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True, # Must add for generation
                enable_thinking = False, # Disable thinking
            )
            model_inputs = tokenizer(text, return_tensors = "pt").to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=1024,
                                            temperature = 0.7, top_p = 0.8, top_k = 20) # For non thinking
            # get the string
            generated_str = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # use the regex to extract the thinking content and the output content
            pattern = re.compile(r"<think>(.*?)</think>(.*)", flags=re.S)
            match = pattern.search(generated_str)
            if match:
                thinking_content = match.group(1).strip()
                output_content = match.group(2).strip()
            else:
                print("Regex failed")
                continue

            # write the response to the output file
            return_item = {
                'input': input_text,
                'output': output_content,
                'output_with_context': input_text + "\n\n" + thinking_content + "\n\n" + output_content,
                'answer': reference_text,
            }
            fout.write(json.dumps(return_item) + '\n')

            count += 1
            if count >= TEST_NUM:
                break
            elif count % 10 == 0:
                print(f"The {count}/{TEST_NUM} items are processed")
                print(f"The answer of the {count}th item is:")
                print(output_content)
                print(f"The ground truth of the {count}th item is:")
                print(reference_text)
                print("------------------------------------------------------------------------------------------------")