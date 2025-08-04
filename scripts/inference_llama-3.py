from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import argparse
import json
import os

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None
load_in_4bit = False

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
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
    parser.add_argument(
        "--test_num",
        type=int,
        default=2000,
        help="The number of items to test"
    )
    parser.add_argument(
        "--print_every_10_items",
        action="store_true",
        help="Print the result every 10 items"
    )

    args = parser.parse_args()

    # create the output directory if it doesn't exist
    dirpath = os.path.dirname(args.output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    # Get the tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template='llama-3',  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
    )

    # Start inference
    count = 0
    print(f"Start inference, model: {args.model_name}, data: {args.data_path}, output: {args.output_path}")
    print("------------------------------------------------------------------------------------------------")

    with open(args.data_path, 'r') as fin, open(args.output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            input_text = item['conversations'][0]['value']
            reference_text = item['conversations'][1]['value']
            messages = [
                {"from": "human",
                "value": input_text},
            ]
            # prepare the input ids
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Must add for generation
                return_tensors="pt",
            ).to("cuda")
            input_len = input_ids.shape[-1]
            answer_ids = model.generate(input_ids=input_ids, max_new_tokens=512, use_cache=True,
                                        pad_token_id=tokenizer.eos_token_id)

            # get the generated ids
            gen_ids = answer_ids[:, input_len:]
            gen_str = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            answer_str = tokenizer.batch_decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

            # write the response to the output file
            return_item = {
                'input': input_text,
                'output': gen_str,
                'output_with_context': answer_str,
                'answer': reference_text,
            }
            fout.write(json.dumps(return_item) + '\n')

            count += 1
            if count >= args.test_num:
                break
            elif count % 10 == 0 and args.print_every_10_items:
                print(f"The {count}/{args.test_num} items are processed")
                print(f"The answer of the {count}th item is:")
                print(gen_str)
                print(f"The ground truth of the {count}th item is:")
                print(reference_text)
                print("------------------------------------------------------------------------------------------------")