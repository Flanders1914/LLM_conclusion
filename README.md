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
│   ├── select_top_results.py       # Extract top/bottom examples by metric
│   ├── formatting.py               # Data formatting utilities
│   ├── filter.py                   # Data filtering utilities
│   ├── Process_PubMedRCT.py        # PubMed RCT data processing
│   ├── Process_PubMed_NonRCT.py    # PubMed non-RCT data processing
│   ├── combine_training_data.py    # Combine RCT and non-RCT training splits
│   └── counting_item_num.py        # Data counting utilities
├── plot/                           # Plotting utilities
│   ├── plot_distribution.py        # Histograms for a metric distribution
│   └── plot_baseline.py            # Bar charts and comprehensive results table
├── output/                         # Model outputs and results
│   ├── models/                     # Fine-tuned model checkpoints
│   ├── predictions/                # output/predictions/{MODEL}/prompt{i}/{rct|non_rct}.jsonl
│   ├── eval_results/               # output/eval_results/{MODEL}/prompt{i}/{split}_results.jsonl
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

### Running the Data Processing Pipeline

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

## Experiments

### Non-Fine-tuning Baseline

For baseline experiments without fine-tuning, we test **Llama-3.1-8B-Instruct** and **Llama-3.2-3B-Instruct**'s performance using all 4 prompt types:

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
For inference on RCT dataset(using Llama-3.2-3B-Instruct):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_llama-3.py \
    --model_name  unsloth/Llama-3.2-3B-Instruct \
    --data_path data/formatted_sharegpt/rct/test.jsonl \
    --output_path output/predictions/llama3.2-3b-base/prompt0/rct.jsonl \
    --test_num 2000 \
    --print_every_10_items
```

For inference on Non-RCT dataset(using Llama-3.2-3B-Instruct):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_llama-3.py \
  --model_name unsloth/Llama-3.2-3B-Instruct \
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

### SFT Experiment

The SFT experiment is conducted on RCT and non-RCT datasets separately.
The base model is `unsloth/Meta-Llama-3.1-8B-Instruct`/ `unsloth/Llama-3.2-3B-Instruct` 
We fine-tune this model on the train split.
For RCT, our experiment will choose different data sizes (10,000, 50,000, 100,000) to show how the results behave as the training data scales.

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
    --output_path output/models/prompt1/llama-3.2-3b-rct50k
```

After fine-tuning the model, perform inference and evaluate the results:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sft_inference.py \
    --saved_path output/models/prompt1/llama-3.2-3b-rct10k/lora_model \
    --data_path data/formatted_sharegpt/non_rct/test.jsonl \
    --output_path output/predictions/llama-3.2-3b-rct10k/prompt1/non_rct.jsonl \
    --test_num 2000

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
    output/eval_results/llama3.1-8b-base/prompt1/rct_results.jsonl \
    output/eval_results/llama3.1-8b-rct10k/prompt1/rct_results.jsonl \
    output/eval_results/llama3.1-8b-rct50k/prompt1/rct_results.jsonl \
  --labels "llama3.1-8b" "llama3.1-8b-rct10k" "llama3.1-8b-rct50k" \
  --output_dir output/figures/compare_3.1/prompt1-rct-rougeL \
  --plot_metric rougeL \
  --bin_size 50
```

```bash
  python plot/plot_distribution.py \
  --input_paths \
    output/eval_results/llama3.2-3b-base/prompt1/rct_results.jsonl \
    output/eval_results/llama3.2-3b-rct10k/prompt1/rct_results.jsonl \
    output/eval_results/llama3.2-3b-rct50k/prompt1/rct_results.jsonl \
  --labels "llama3.2-3b" "llama3.2-3b-rct10k" "llama3.2-3b-rct50k" \
  --output_dir output/figures/compare_3.1/prompt1-rct-rougeL \    
  --plot_metric rougeL \
  --bin_size 50
```

Create baseline bar charts across prompts and a comprehensive table (reads JSONL reports produced by evaluate.sh):

```bash
python plot/plot_baseline.py \
  --eval_root output/eval_results \
  --model_name llama3.1-8b-base \
  --out_dir output/figures/llama3.1-8b-base
```

```bash
python plot/plot_baseline.py \
  --eval_root output/eval_results \
  --model_name llama3.2-3b-base \
  --out_dir output/figures/llama3.2-3b-base
```

### Notes and Tips
- `evaluate.sh` expects predictions under `output/predictions/{MODEL}/prompt{i}/{split}.jsonl`.