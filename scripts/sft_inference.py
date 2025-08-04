from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import argparse
import json
import os

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for the saved finetuned model, only support sharegpt style for now"
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        required=True,
        help="The path to the saved finetuned model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to the data to use"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to the output file"
    )
    parser.add_argument(
        "--is_4bit",
        action='store_true',
        help="Whether the model is 4bit"
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=1000,
        help="The number of test items"
    )

    args = parser.parse_args()

    # create the output directory if it doesn't exist
    dirpath = os.path.dirname(args.output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.saved_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = args.is_4bit,
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
    print(f"Start inference, model: {args.saved_path}, data: {args.data_path}, output: {args.output_path}")
    print("------------------------------------------------------------------------------------------------")

    with open(args.data_path, 'r') as fin, open(args.output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            input_text = item['conversations'][0]['value']
            reference_text = item['conversations'][1]['value']

            # add prompt postfix
            input_text = input_text
            messages = [
                {"from": "human",
                "value": input_text},
            ]
            # prepare the input ids
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
            ).to("cuda")
            input_len = input_ids.shape[-1]
            answer_ids = model.generate(input_ids=input_ids, max_new_tokens=512, use_cache=True)

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
            elif count % 10 == 0:
                print(f"The {count}/{args.test_num} items are processed")
                print(f"The answer of the {count}th item is:")
                print(gen_str)
                print(f"The ground truth of the {count}th item is:")
                print(reference_text)
                print("------------------------------------------------------------------------------------------------")