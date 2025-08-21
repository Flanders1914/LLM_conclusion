# -*- coding: utf-8 -*-
import os
import argparse
import sys


from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
import torch
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

TRAIN_EVAL_RATIO = 0.9
SHUFFLE_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 4
OPTIMIZER = "adamw_8bit"
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
SAVE_STEPS = 1000
SCHEDULER_TYPE = "linear"
LOGGING_STEPS = 10

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
    parser.add_argument('--load_in_4bit', action='store_true', help='Use 4bit quantization to reduce memory usage.')

    # training
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size, must be divisible by 4(GRADIENT_ACCUMULATION_STEPS)')
    parser.add_argument('--num_epoch', default=10, type=int, help='number of epochs')
    parser.add_argument('--max_eval_samples', default=100, type=int, help='maximum number of samples for evaluation')
    parser.add_argument('--eval_steps', default=500, type=int, help='number of steps for evaluation')

    # output
    parser.add_argument('--output_path', required=True, type=str, help='output path')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':

    args = get_argument()

    # get batch size per device
    if args.batch_size % GRADIENT_ACCUMULATION_STEPS != 0:
        print(f"Batch size must be divisible by {GRADIENT_ACCUMULATION_STEPS}, exiting...")
        sys.exit(1)
    batch_size_per_device = args.batch_size // GRADIENT_ACCUMULATION_STEPS

    # create output path if not exists
    parent = os.path.dirname(args.output_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    # if 4bit model, set load_in_4bit to True
    if '4bit' in args.model:
        args.load_in_4bit = True
    else:
        args.load_in_4bit = False

    # get device
    if not torch.cuda.is_available():
        print("No GPU available, exiting...")
        sys.exit(1)
    device = torch.cuda.current_device()
    print(f"Using device: {device}")

    # load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,  # Choose ANY!
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        device_map = "auto"
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
        use_gradient_checkpointing=True,  # True or "unsloth" for very long context
        random_state=args.seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
   
    # prepare dataset
    dataset = load_dataset("json", data_files=args.data_path)["train"]
    dataset = dataset.select_columns(["conversations"])
    print(f"The loaded dataset size is: {len(dataset)}")

    # split dataset, set seed to ensure reproducibility
    splits = dataset.train_test_split(train_size=TRAIN_EVAL_RATIO, shuffle=True, seed=SHUFFLE_SEED)
    dataset_train = splits["train"]
    dataset_val = splits["test"]
    # check data size
    if args.data_size < len(dataset_train):
        dataset_train = dataset_train.select(range(args.data_size))
    else:
        print(f"The dataset size for training is set to {len(dataset_train)} because there are only {args.data_size} samples in the training set")
    if args.max_eval_samples < len(dataset_val):
        dataset_val = dataset_val.select(range(args.max_eval_samples))
    else:
        print(f"Max eval samples is set to {len(dataset_val)} because there are only {len(dataset_val)} samples in the validation set")


    print(f"Dataset size for training is: {len(dataset_train)}")
    print(f"Dataset size for validation is: {len(dataset_val)}")
    print(f"Every {args.eval_steps} steps, we will evaluate the model on {len(dataset_val)} samples")

    # formatting dataset
    if args.data_format == 'sharegpt':
        if 'Llama' in args.model or 'llama' in args.model:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="llama-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
            )
        else:
            # TODO: support qwen and other models
            print("Only support llama model for now!")
            sys.exit(1)
        # formatting function
        def formatting_sharegpt_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
            return {"text": texts}
        # apply formatting function
        dataset_train = dataset_train.map(formatting_sharegpt_prompts_func, batched=True)
        dataset_val = dataset_val.map(formatting_sharegpt_prompts_func, batched=True)
        # see the first example
        print("The first item of the dataset:")
        print(dataset_train[0])
        print("-" * 100)
    else:
        print("Only support sharegpt format for now!")
        sys.exit(1)

    # create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        dataset_text_field = "text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = batch_size_per_device,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
            warmup_steps = WARMUP_STEPS,
            num_train_epochs = args.num_epoch,
            eval_strategy = "steps",
            eval_steps = args.eval_steps,
            save_steps=SAVE_STEPS,
            learning_rate = args.lr,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = LOGGING_STEPS,
            optim = OPTIMIZER,
            weight_decay = WEIGHT_DECAY,
            lr_scheduler_type = SCHEDULER_TYPE,
            report_to = "none",
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