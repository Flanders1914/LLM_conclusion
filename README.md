# LLM Conclusion Generation

A project for inference the conclusion section in the research paper abstract.

## Project Structure

```
LLM_conclusion/
├── data/                    # Data directories
│   ├── raw/                # Raw PubMed datasets
│   ├── processed/          # Processed data files
│   ├── filtered/           # Filtered datasets
│   └── formatted_sharegpt/ # ShareGPT formatted data
├── scripts/                # Main processing scripts
│   ├── sft.py             # Supervised fine-tuning script
│   ├── sft_inference.py   # Inference script for fine-tuned models
│   ├── inference_llama-3.py # Llama-3 specific inference
│   ├── inference_qwen3.py # Qwen3 specific inference
│   ├── evaluate_result.py # Evaluation metrics calculation
│   ├── formatting.py      # Data formatting utilities
│   ├── filter.py          # Data filtering utilities
│   ├── Process_PubMedRCT.py # PubMed RCT data processing
│   └── Process_PubMed_NonRCT.py # PubMed non-RCT data processing
├── output/                 # Model outputs and results
├── conclusion_env/         # Virtual environment
├── unsloth_compiled_cache/ # Unsloth cache directory
└── requirements.txt        # Python dependencies
```

## Raw data source
### PubMed 20k RCT
The PubMed 200k/20k RCT dataset is from *Franck Dernoncourt, Ji Young Lee. [PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts](https://arxiv.org/abs/1710.06071). International Joint Conference on Natural Language Processing (IJCNLP). 2017.*

This project uses PubMed 20k RCT from this paper.

The git repo of pubMed 20k RCT is https://github.com/Franck-Dernoncourt/pubmed-rct

### PubMed Non-RCT
The PubMed non-RCT corpus is from the paper [**Segmenting Scientific Abstracts into Discourse Categories: A  Deep Learning-Based Approach for Sparse Labeled Data**](https://dl.acm.org/doi/abs/10.1145/3383583.3398598) (  [Arxiv preprint](https://arxiv.org/abs/2005.05414)  ), 	presented in JCDL 2020.

The git repo of PubMed non-RCT is https://github.com/soumyaxyz/abstractAnalysis/blob/master/README.md?plain=1

## Installation

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

## Usage

### 1. Downloading raw data from github
The raw data is on Github.

```bash
# Create data directories
mkdir -p data/raw
cd data/raw
git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git
git clone https://github.com/soumyaxyz/abstractAnalysis.git
```

### 2. Data Processing
This step processes the raw .txt files from both PubMed RCT and non-RCT datasets, extracting the structured data and converting it into JSONL format.

### 3. Data Filtering
This step filters the processed datasets to ensure quality and consistency. The filtering criteria include:
- Must have METHODS and RESULTS labeled sentences
- Must have at least 1 CONCLUSIONS labeled sentence
- CONCLUSIONS labeled sentences must be the last sentences of the item
- The number of CONCLUSIONS labeled sentences must be smaller than the number of METHODS and RESULTS labeled sentences combined

### 4. Data Formatting
This step converts the filtered data into Alpaca or ShareGPT formats and adds prompts.

### 5. SFT (Supervised Fine-Tuning)
This step performs supervised fine-tuning on pre-trained language models using the formatted data. The script uses Unsloth for efficient training with LoRA.

### 6. Reference (Inference)
This step performs inference using the fine-tuned models to generate conclusions from research abstracts.

### 7. Evaluate
This step evaluates the generated conclusions using multiple metrics including ROUGE, BLEU, and METEOR scores.

**Evaluation metrics**:
- **ROUGE**: Measures overlap of n-grams between generated and reference text
- **BLEU**: Measures precision of n-gram matches
- **METEOR**: Combines exact, stem, synonym, and paraphrase matches
- **Word count analysis**: Compares average word count of predictions vs references