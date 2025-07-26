import os
import math
import argparse

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
import torch
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def get_argument():
    parser = argparse.ArgumentParser(description='Description of your program')

    # data
    parser.add_argument('--data_path', required=True, type=str, help='dataset path')
    parser.add_argument('--data_size', required=True, type=int, help='dataset size')
    parser.add_argument('--data_format', default='sharegpt', type=str, help='dataset format')

    # model
    parser.add_argument('--seed', default=3407, type=int, help='random seed for the model')
    parser.add_argument('--model', required=True, type=str, help='model to finetune')
    parser.add_argument('--max_seq_length', default=2048, type=int)
    parser.add_argument('--dtype', default=None, type=str, help='None for auto detection.')
    parser.add_argument('--load_in_4bit', default=False, type=bool, help='Use 4bit quantization to reduce memory usage.')

    # training
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=10, type=int, help='number of epochs')

    # output
    parser.add_argument('--output_path', required=True, type=str, help='output path')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':

    args = get_argument()

    # create output path if not exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # if 4bit model, set load_in_4bit to True
    if '4bit' in args.model:
        args.load_in_4bit = True
    else:
        args.load_in_4bit = False

    # load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,  # Choose ANY!
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    # apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=args.seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
   
    # prepare dataset
    dataset = load_dataset("json", data_files=args.data_path)["train"]
    if args.data_size < len(dataset):
        dataset = dataset.select(range(args.data_size))
    print(f"Dataset size for training is: {len(dataset)}")

    if args.data_format == 'sharegpt':
        if 'llama' in args.model:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="llama-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
            )
        else:
            # TODO: support qwen and other models
            print("Only support llama model for now!")
            exit()
        # formatting function
        def formatting_sharegpt_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
            return {"text": texts}
        # apply formatting function
        dataset = dataset.map(formatting_sharegpt_prompts_func, batched=True)
        # see the first example
        print("The first item of the dataset:")
        print(dataset[0])
        print("-" * 100)
    else:
        print("Only support sharegpt format for now!")
        exit()

    # create trainer
    total_steps = args.num_epoch * math.ceil(len(dataset) / args.batch_size)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field = "text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = total_steps,
            learning_rate = args.lr,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = args.seed,
            output_dir=os.path.join(args.output_path, 'checkpoints'),
        ),
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # train the model
    print("-" * 100)
    print("-" * 100)
    print("Start training the model...")
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained(os.path.join(args.output_path, "lora_model"))
    tokenizer.save_pretrained(os.path.join(args.output_path, "lora_model"))