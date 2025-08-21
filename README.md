# LLM Conclusion Generation

A project for inferring the conclusion section in research paper abstracts using Large Language Models.

## Project Structure

```
LLM_conclusion/
├── data/                           # Data directories
│   ├── raw/                        # Raw PubMed datasets
│   ├── processed/                  # Processed data files
│   ├── filtered/                   # Filtered datasets
│   ├── formatted_sharegpt/         # ShareGPT formatted data
│   │   ├── rct/                    # RCT dataset splits
│   │   │   ├── train.jsonl         # Training data (168,008 entries)
│   │   │   ├── dev.jsonl           # Development data (2,223 entries)
│   │   │   └── test.jsonl          # Test data (2,169 entries)
│   │   └── non_rct/                # Non-RCT dataset splits
│   │       ├── train.jsonl         # Training data (14,539 entries)
│   │       ├── dev.jsonl           # Development data (2,409 entries)
│   │       └── test.jsonl          # Test data (2,406 entries)
│   └── formatted_alpaca/           # Optional: Alpaca formatted data (if --format alpaca)
│       ├── rct/
│       └── non_rct/
├── scripts/                        # Main processing and experiment scripts
│   ├── sft.py                      # Supervised fine-tuning
│   ├── sft_inference.py            # Inference for fine-tuned models
│   ├── inference_llama-3.py        # Llama-3 specific inference
│   ├── inference_qwen3.py          # Qwen3 specific inference
│   ├── evaluate_result.py          # Metrics calculation (ROUGE/BLEU/METEOR/word counts)
│   ├── evaluate_with_CLAIR.py      # Semantic similarity evaluation using ChatGPT/GPT models
│   ├── select_top_results.py       # Extract top/bottom examples by metric
│   ├── formatting.py               # Data formatting utilities
│   ├── filter.py                   # Data filtering utilities
│   ├── Process_PubMedRCT.py        # PubMed RCT data processing
│   ├── Process_PubMed_NonRCT.py    # PubMed non-RCT data processing
│   └── counting_item_num.py        # Data counting utilities
├── plot/                           # Plotting utilities
│   ├── plot_distribution.py        # Histograms for a metric distribution
│   ├── plot_accumulative.py        # Cumulative distribution curves for metrics
│   └── plot_baseline.py            # Bar charts and comprehensive results table
├── output/                         # Model outputs and results
│   ├── models/                     # Fine-tuned model checkpoints
│   ├── predictions/                # output/predictions/{MODEL}/prompt{i}/{rct|non_rct}.jsonl
│   ├── eval_results/               # output/eval_results/{MODEL}/prompt{i}/{split}_results.jsonl
│   ├── LLM_evals/                  # CLAIR evaluation results using ChatGPT/GPT models
│   └── figures/                    # Saved figures and tables
├── evaluate.sh                     # Batch evaluation helper across prompts
├── run_data_processing.sh          # Data processing pipeline script
├── requirements.txt                # Python dependencies
├── README.md
├── conclusion_env/                 # Virtual environment (local)
├── unsloth_compiled_cache/         # Unsloth cache directory
└── .gitignore                      # Git ignore file
```

## Raw Data Sources

### PubMed 200k RCT
The PubMed 200k/20k RCT dataset is from *Franck Dernoncourt, Ji Young Lee. [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071). International Joint Conference on Natural Language Processing (IJCNLP). 2017.*

This project uses PubMed 200k RCT from this paper.

The git repository of PubMed 200k RCT is: https://github.com/Franck-Dernoncourt/pubmed-rct

### PubMed Non-RCT
The PubMed non-RCT corpus is from the paper [**Segmenting Scientific Abstracts into Discourse Categories: A Deep Learning-Based Approach for Sparse Labeled Data**](https://dl.acm.org/doi/abs/10.1145/3383583.3398598) ([Arxiv preprint](https://arxiv.org/abs/2005.05414)), presented in JCDL 2020.

The git repository of PubMed non-RCT is: https://github.com/soumyaxyz/abstractAnalysis/blob/master/README.md?plain=1

### CNN/DailyMail
The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization, though the original version was created for machine reading and comprehension and abstractive question answering.
Versions 2.0.0 and 3.0.0 of the CNN / DailyMail Dataset can be used to train a model for abstractive and extractive summarization (Version 1.0.0 was developed for machine reading and comprehension and abstractive question answering).

## Usage

### Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLM_conclusion
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv conclusion_env
   source conclusion_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download raw data from GitHub**:
   ```bash
   # Create data directories
   mkdir -p data/raw
   cd data/raw
   git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git
   git clone https://github.com/soumyaxyz/abstractAnalysis.git
   ```

5. **Extract RCT train split**:
   ```bash
   7z x data/raw/pubmed-rct/PubMed_200k_RCT/train.7z -o./data/raw/pubmed-rct/PubMed_200k_RCT/
   ```

6. **Download and Prepare CNN/Dailymail Dataset**
  The prepare_summarization_data.py automatically prepare the version 3.0.0 CNN/dailymail dataset as the sharegpt format, and store the data in data/formatted_sharegpt/cnn_dailymail/ directory in jsonl format.

  Note: This script uses **"Given the above text, please write a summary in the format of CNN article highlights. The summary is:"** as the prompt.
   ```bash
   python scripts/prepare_summarization_data.py
   ```
   
### Running the Data Processing Pipeline
run_data_processing.sh script is the universal utility to process the dataset of RCT-200k and Non-RCT.
It supports formating the data with 4 different prompts in sharegpt format. 

```bash
./run_data_processing.sh
```

The `run_data_processing.sh` script can also run in formatting mode, which is important for using different prompts.
See:
```bash
./run_data_processing.sh -h
```

This pipeline includes the following steps:

#### 1. Data Processing
This step processes the raw .txt files from both PubMed RCT and non-RCT datasets, extracting the structured data and converting it into JSONL format.

#### 2. Data Filtering
This step filters the processed datasets to ensure quality and consistency. The filtering criteria include:
- Must have METHODS and RESULTS labeled sentences
- Must have at least 1 CONCLUSIONS labeled sentence
- CONCLUSIONS labeled sentences must be the last sentences of the item
- The number of CONCLUSIONS labeled sentences must be smaller than the number of METHODS and RESULTS labeled sentences combined

#### 3. Data Formatting
This step converts the filtered data into Alpaca or ShareGPT formats and adds prompts.

After processing, the data will be organized as follows:

### RCT Dataset (`data/formatted_sharegpt/rct/`)
| File | Number of Entries |
|------|------------------|
| train.jsonl | 168,008 |
| dev.jsonl | 2,223 |
| test.jsonl | 2,169 |

### Non-RCT Dataset (`data/formatted_sharegpt/non_rct/`)
| File | Number of Entries |
|------|------------------|
| train.jsonl | 14,539 |
| dev.jsonl | 2,409 |
| test.jsonl | 2,406 |

Each data entry follows this JSON format:
```json
{
    "id": "...", // the PubMed id of the paper
    "title": "...", // the title of the paper
    "sentences": [{"label": "OBJECTIVE", "text": "......."}], // each obj has 'label' and 'text' attributes
    "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
    ]
}
```

## Experiments for RCT and Non-RCT

### Non-Fine-tuning Baseline

For baseline experiments without fine-tuning, we test **Llama3.1-8B-Instruct** and **Llama3.2-3B-Instruct**'s performance using all 4 prompt types:

- **Prompt 0**: "Given the above text, please write a conclusion section. The conclusion section is:"
- **Prompt 1**: "Given the above text, please write a conclusion section in the format of PubMed paper abstract. The conclusion section is:"
- **Prompt 2**: "Given the above text, please write a {} sentences conclusion section. The conclusion section is:"
- **Prompt 3**: "Given the above text, please write a {} sentences conclusion section in the format of PubMed paper abstract. The conclusion section is:"

We use the first 2,000 data entries of the test split in both RCT and non-RCT datasets.

#### Step 1: Formatting Test Data with the Specific Prompt
```bash
./run_data_processing.sh --formatting-only -p 0
```

#### Step 2: Running inference_llama-3.py
For inference on RCT dataset(using Llama3.2-3B-Instruct):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_llama-3.py \
    --model_name  unsloth/Llama3.2-3B-Instruct \
    --data_path data/formatted_sharegpt/rct/test.jsonl \
    --output_path output/predictions/llama3.2-3b-base/prompt0/rct.jsonl \
    --test_num 2000 \
    --print_every_10_items
```

For inference on Non-RCT dataset(using Llama-3.2-3B-Instruct):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_llama-3.py \
  --model_name unsloth/Llama3.2-3B-Instruct \
  --data_path data/formatted_sharegpt/non_rct/test.jsonl \
  --output_path output/predictions/llama3.2-3b-base/prompt0/non_rct.jsonl \
  --test_num 2000 \
  --print_every_10_items
```

#### Step 3: Repeat Steps 1-2 for Other Prompts
To format the input with other prompts indexed by i (where i ∈ [0, 1, 2, 3]):
```bash
./run_data_processing.sh --formatting-only -p i
```

#### Step 4: Evaluate the results
You can evaluate all prompts for a given model directory using the helper script:

```bash
# First edit the evaluate.sh to choose the prediction directory
./evaluate.sh
```

#### Step 5: LLM Evaluation
For advanced semantic similarity evaluation using ChatGPT/GPT models, you can use the CLAIR evaluation script:

```bash
python scripts/evaluate_with_LLM.py \
    --input output/predictions/llama3.2-3b-rct50k/prompt1/rct.jsonl \
    --api-key ***************** \
    --output output/LLM_evals/gpt3.5/llama3.2-3b-rct50k/prompt1/rct.jsonl \
    --model gpt-3.5-turbo \
    --delay 0.5
```

**CLAIR Evaluation Parameters:**
- `--input`: Path to JSONL file with prediction results (expects 'output' and 'answer' fields)
- `--api-key`: Your OpenAI API key (should start with 'sk-')
- `--output`: Output JSON file path for CLAIR evaluation results
- `--model`: OpenAI model to use (default: 'gpt-3.5-turbo', also supports 'gpt-4', etc.)
- `--max-pairs`: Maximum number of conclusion pairs to evaluate (optional, evaluates all if not specified)
- `--delay`: Delay between API calls in seconds (default: 1.0, adjust based on your API rate limits)

### SFT Experiment
The SFT experiment is conducted on RCT and non-RCT datasets separately.
The base model is `unsloth/Llama-3.2-3B-Instruct` 
We fine-tune this model on the train split.
For RCT, our experiment will choose different data sizes (10,000, 50,000) to show how the results behave as the training data scales.

Here is an SFT example:
```bash
./run_data_processing.sh --formatting-only -p 1

CUDA_VISIBLE_DEVICES=0 python scripts/sft.py \
    --data_path ./data/formatted_sharegpt/rct/train.jsonl \
    --data_size 50000 \
    --seed 3407 \
    --data_format sharegpt \
    --model unsloth/Llama-3.2-3B-Instruct \
    --max_seq_length 2048 \
    --lr 1e-4 \
    --batch_size 16 \
    --num_epoch 1 \
    --max_eval_samples 100 \
    --eval_steps 400 \
    --output_path output/models/prompt1/llama3.2-3b-rct50k
```

After fine-tuning the model, perform inference and evaluate the results:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sft_inference.py \
    --saved_path output/models/prompt1/llama3.2-3b-rct10k/lora_model \
    --data_path data/formatted_sharegpt/non_rct/test.jsonl \
    --output_path output/predictions/llama3.2-3b-rct10k/prompt1/non_rct.jsonl \
    --test_num 2000 \
    --max_seq_length 2048
```

Changing the prediction directory in evaluate.sh then:
```bash
./evaluate.sh
```

### Plotting Figures

Plot distributions for a metric across all results:
```bash
python plot/plot_distribution.py \
  --input_paths \
    output/eval_results/llama3.2-3b-base/prompt1/rct_results.jsonl \
    output/eval_results/llama3.2-3b-rct10k/prompt1/rct_results.jsonl \
    output/eval_results/llama3.2-3b-rct50k/prompt1/rct_results.jsonl \
  --labels "llama3.2-3b" "llama3.2-3b-rct10k" "llama3.2-3b-rct50k" \
  --output_dir output/figures/compare_3.2/prompt1-rct-rougel \
  --plot_metric rougeL \
  --bin_size 50
```

Create accumulative (cumulative distribution) curves:

```bash
python plot/plot_accumulative.py \
  --input_paths \
    output/eval_results/llama3.2-3b-base/prompt1/rct_results.jsonl \
    output/eval_results/llama3.2-3b-rct10k/prompt1/rct_results.jsonl \
    output/eval_results/llama3.2-3b-rct50k/prompt1/rct_results.jsonl \
  --labels "llama3.2-3b" "llama3.2-3b-rct10k" "llama3.2-3b-rct50k" \
  --output_dir output/figures/compare_3.2/prompt1-rct-cdf \
  --plot_metric word_count_prediction \
  --num_points 1000
```

Create baseline bar charts across prompts and a comprehensive table (reads JSONL reports produced by evaluate.sh):

```bash
python plot/plot_baseline.py \
  --eval_root output/eval_results \
  --model_name llama3.2-3b-base \
  --out_dir output/figures/llama3.2-3b-base
```

### Selecting top results
```bash
python scripts/select_top_results.py \
--input_path output/eval_results/llama3.2-3b-base/prompt1/rct_results.jsonl \
--output_path output/eval_results/llama3.2-3b-base/prompt1/rct_results_top_5_meteor.json \
--selection_metric meteor \
--top_n 5 \
>> output/eval_results/llama3.2-3b-base/prompt1/rct_results_top_5_meteor.txt
```

### Notes and Tips
- Ensure **using the same prompt** for finetuning and inference.
- `./run_data_processing.sh --formatting-only -p i` to format all data in a chosen prompt
- `evaluate.sh` expects predictions under `output/predictions/{MODEL}/prompt{i}/{split}.jsonl`.
- Rename the output path correctly according to which prompt is used and the model

## Experiments for CNN/dailymail
Set `test_num` to 99999 (larger than the number of records in test split) for testing all records 

Note: for CNN/dailymail task, max_seq_length=2048(default) is not enough, set it to 16384 or lager

### Baseline inference
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_llama-3.py \
    --model_name  unsloth/Llama3.2-3B-Instruct \
    --data_path data/formatted_sharegpt/cnn_dailymail/test.jsonl\
    --output_path output/predictions/cnn_dailymail/llama3.2-3b-base.jsonl \
    --test_num 99999 \
    --print_every_10_items \
    --max_seq_length 16384
```
### FineTuning
**Choose any data_size**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sft.py \
    --data_path data/formatted_sharegpt/cnn_dailymail/train.jsonl \
    --data_size 20000 \
    --seed 3407 \
    --data_format sharegpt \
    --model unsloth/Llama-3.2-3B-Instruct \
    --max_seq_length 16384 \
    --lr 1e-4 \
    --batch_size 16 \
    --num_epoch 1 \
    --max_eval_samples 100 \
    --eval_steps 400 \
    --output_path output/models/cnn_dailymail/llama3.2-3b-20k
```
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sft_inference.py \
    --saved_path output/models/cnn_dailymail/llama3.2-3b-20k \
    --data_path data/formatted_sharegpt/cnn_dailymail/test.jsonl \
    --output_path output/predictions/cnn_dailymail/llama3.2-3b-20k.jsonl \
    --test_num 99999 \
    --max_seq_length 16384
```
