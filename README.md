# LLM Conclusion Generation

A project for inferring the conclusion section in research paper abstracts using Large Language Models.

## Project Structure

```
LLM_conclusion/
├── data/                    # Data directories
│   ├── raw/                # Raw PubMed datasets
│   ├── processed/          # Processed data files
│   ├── filtered/           # Filtered datasets
│   └── formatted_sharegpt/ # ShareGPT formatted data
│       ├── rct/            # RCT dataset splits
│       │   ├── train.jsonl # Training data (168,008 entries)
│       │   ├── dev.jsonl   # Development data (2,223 entries)
│       │   └── test.jsonl  # Test data (2,169 entries)
│       └── non_rct/        # Non-RCT dataset splits
│           ├── train.jsonl # Training data (14,539 entries)
│           ├── dev.jsonl   # Development data (2,409 entries)
│           └── test.jsonl  # Test data (2,406 entries)
├── scripts/                # Main processing scripts
│   ├── sft.py             # Supervised fine-tuning script
│   ├── sft_inference.py   # Inference script for fine-tuned models
│   ├── inference_llama-3.py # Llama-3 specific inference
│   ├── inference_qwen3.py # Qwen3 specific inference
│   ├── evaluate_result.py # Evaluation metrics calculation
│   ├── formatting.py      # Data formatting utilities
│   ├── filter.py          # Data filtering utilities
│   ├── Process_PubMedRCT.py # PubMed RCT data processing
│   ├── Process_PubMed_NonRCT.py # PubMed non-RCT data processing
│   ├── combine_training_data.py # Training data combination utilities
│   └── counting_item_num.py # Data counting utilities
├── output/                 # Model outputs and results
│   ├── models/            # Fine-tuned model checkpoints
│   └── predictions/       # Model prediction outputs
├── conclusion_env/         # Virtual environment
├── unsloth_compiled_cache/ # Unsloth cache directory
├── run_data_processing.sh  # Main data processing pipeline script
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore file
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

For baseline experiments without fine-tuning, we test Meta-Llama-3.1-8B-Instruct's performance using all 4 prompt types:

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
For inference on RCT dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_llama-3.py \
    --model_name unsloth/Meta-Llama-3.1-8B-Instruct \
    --data_path data/formatted_sharegpt/rct/test.jsonl \
    --output_path output/predictions/llama3_8b_base/rct.jsonl \
    --test_num 2000 \
    --print_every_10_items
```

For inference on Non-RCT dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_llama-3.py \
    --model_name unsloth/Meta-Llama-3.1-8B-Instruct \
    --data_path data/formatted_sharegpt/non_rct/test.jsonl \
    --output_path output/predictions/llama3_8b_base/non_rct.jsonl \
    --test_num 2000 \
    --print_every_10_items
```

#### Step 3: Evaluate the Results
```bash
python scripts/evaluate_result.py \
    --input_path output/predictions/llama3_8b_base/non_rct.jsonl \
    --output_path output/predictions/llama3_8b_base/non_rct_result.json

python scripts/evaluate_result.py \
    --input_path output/predictions/llama3_8b_base/rct.jsonl \
    --output_path output/predictions/llama3_8b_base/rct_result.json
```

#### Step 4: Repeat Steps 1-3 for Other Prompts
To format the input with other prompts indexed by i (where i ∈ [0, 1, 2, 3]):
```bash
./run_data_processing.sh --formatting-only -p i
```

### SFT Experiment

The SFT experiment is conducted on RCT and non-RCT datasets separately.
The base model is `unsloth/Meta-Llama-3.1-8B-Instruct`.
We fine-tune this model on the train split.
For RCT, our experiment will choose different data sizes (5,000, 10,000, 50,000, 100,000) to show how the results behave as the training data scales.

Here is an SFT example:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/sft.py \
    --data_path ./data/formatted_sharegpt/rct/train.jsonl \
    --data_size 5000 \
    --seed 3407 \
    --data_format sharegpt \
    --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --max_seq_length 2048 \
    --lr 1e-4 \
    --batch_size 16 \
    --num_epoch 3 \
    --max_eval_samples 100 \
    --eval_steps 500 \
    --output_path output/models/llama3-8b-rct5000
```

After fine-tuning the model, perform inference and evaluate the results:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/sft_inference.py \
    --saved_path output/models/llama3-8b-rct5000/lora_model \
    --data_path data/formatted_sharegpt/rct/test.jsonl \
    --output_path output/predictions/llama3-8b-rct5000/rct.jsonl \
    --test_num 2000

python scripts/evaluate_result.py \
    --input_path output/predictions/llama3-8b-rct5000/rct.jsonl \
    --output_path output/predictions/llama3-8b-rct5000/rct_result.json
```
