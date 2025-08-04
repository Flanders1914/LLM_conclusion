#!/bin/bash

# LLM Conclusion Generation - Data Processing Pipeline
# This script automates the complete data processing workflow

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROMPT_INDEX=0
FORMAT_TYPE="sharegpt"
SKIP_PROCESSING=false
SKIP_FILTERING=false
SKIP_FORMATTING=false
RUN_FORMATTING_ONLY=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -p, --prompt-index INDEX    Prompt index to use (0-3, default: 0)"
    echo "  -f, --format FORMAT         Output format: alpaca or sharegpt (default: sharegpt)"
    echo "  --skip-processing           Skip data processing step"
    echo "  --skip-filtering            Skip data filtering step"
    echo "  --skip-formatting           Skip data formatting step"
    echo "  --formatting-only           Run only the formatting step (requires filtered data)"
    echo "  -h, --help                  Show this help message"
    echo
    echo "Prompt indices:"
    echo "  0: Given the above text, please write a conclusion section. The conclusion section is:"
    echo "  1: Given the above text, please write a conclusion section in the format of PubMed paper abstract. The conclusion section is:"
    echo "  2: Given the above text, please write a {N} sentences conclusion section. The conclusion section is:"
    echo "  3: Given the above text, please write a {N} sentences conclusion section in the format of PubMed paper abstract. The conclusion section is:"
    echo
    echo "Examples:"
    echo "  $0                                    # Run complete pipeline with default settings"
    echo "  $0 -p 2                              # Run with prompt index 2"
    echo "  $0 --formatting-only -p 1            # Run only formatting with prompt index 1"
    echo "  $0 --skip-processing --skip-filtering # Run only formatting step"
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--prompt-index)
                PROMPT_INDEX="$2"
                shift 2
                ;;
            -f|--format)
                FORMAT_TYPE="$2"
                shift 2
                ;;
            --skip-processing)
                SKIP_PROCESSING=true
                shift
                ;;
            --skip-filtering)
                SKIP_FILTERING=true
                shift
                ;;
            --skip-formatting)
                SKIP_FORMATTING=true
                shift
                ;;
            --formatting-only)
                RUN_FORMATTING_ONLY=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Validate prompt index
    if [[ ! "$PROMPT_INDEX" =~ ^[0-3]$ ]]; then
        print_error "Invalid prompt index: $PROMPT_INDEX. Must be 0, 1, 2, or 3."
        exit 1
    fi

    # Validate format type
    if [[ "$FORMAT_TYPE" != "alpaca" && "$FORMAT_TYPE" != "sharegpt" ]]; then
        print_error "Invalid format: $FORMAT_TYPE. Must be 'alpaca' or 'sharegpt'."
        exit 1
    fi
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Virtual environment not detected. Please activate your virtual environment first:"
        echo "source conclusion_env/bin/activate"
        exit 1
    fi
}

# Function to check if required data directories exist
check_data_directories() {
    print_status "Checking data directories..."
    
    if [[ "$RUN_FORMATTING_ONLY" == "false" && "$SKIP_PROCESSING" == "false" ]]; then
        # Check if raw data exists
        if [[ ! -d "data/raw/pubmed-rct" ]]; then
            print_error "PubMed RCT data not found. Please run the data download step first:"
            echo "mkdir -p data/raw && cd data/raw"
            echo "git clone https://github.com/Franck-Dernoncourt/pubmed-rct.git"
            exit 1
        fi

        if [[ ! -d "data/raw/abstractAnalysis" ]]; then
            print_error "PubMed Non-RCT data not found. Please run the data download step first:"
            echo "mkdir -p data/raw && cd data/raw"
            echo "git clone https://github.com/soumyaxyz/abstractAnalysis.git"
            exit 1
        fi
    fi
    
    print_success "Data directories check passed"
}

# Function to create output directories
create_directories() {
    print_status "Creating output directories..."
    
    mkdir -p data/processed/rct
    mkdir -p data/processed/non_rct
    mkdir -p data/filtered/rct
    mkdir -p data/filtered/non_rct
    
    if [[ "$FORMAT_TYPE" == "sharegpt" ]]; then
        mkdir -p data/formatted_sharegpt/rct
        mkdir -p data/formatted_sharegpt/non_rct
    else
        mkdir -p data/formatted_alpaca/rct
        mkdir -p data/formatted_alpaca/non_rct
    fi
    
    print_success "Output directories created"
}

# Function to process PubMed RCT data
process_pubmed_rct() {
    print_status "Processing PubMed RCT data..."
    
    # Process dev, test, and train files
    python scripts/Process_PubMedRCT.py ./data/processed/rct dev.txt
    python scripts/Process_PubMedRCT.py ./data/processed/rct test.txt
    python scripts/Process_PubMedRCT.py ./data/processed/rct train.txt
    
    print_success "PubMed RCT data processing completed"
}

# Function to process PubMed Non-RCT data
process_pubmed_non_rct() {
    print_status "Processing PubMed Non-RCT data..."
    
    # Process train_clean, dev_clean, and test_clean files
    python scripts/Process_PubMed_NonRCT.py ./data/processed/non_rct train_clean.txt
    python scripts/Process_PubMed_NonRCT.py ./data/processed/non_rct dev_clean.txt
    python scripts/Process_PubMed_NonRCT.py ./data/processed/non_rct test_clean.txt
    
    print_success "PubMed Non-RCT data processing completed"
}

# Function to filter processed data
filter_data() {
    print_status "Filtering processed data..."
    
    # Filter PubMed RCT data
    python scripts/filter.py data/processed/rct/dev.jsonl data/filtered/rct
    python scripts/filter.py data/processed/rct/test.jsonl data/filtered/rct
    python scripts/filter.py data/processed/rct/train.jsonl data/filtered/rct
    
    # Filter PubMed Non-RCT data
    python scripts/filter.py data/processed/non_rct/dev_clean.jsonl data/filtered/non_rct
    python scripts/filter.py data/processed/non_rct/test_clean.jsonl data/filtered/non_rct
    python scripts/filter.py data/processed/non_rct/train_clean.jsonl data/filtered/non_rct
    
    print_success "Data filtering completed"
}

# Function to format data
format_data() {
    print_status "Formatting data into $FORMAT_TYPE format with prompt index $PROMPT_INDEX..."
    
    if [[ "$FORMAT_TYPE" == "sharegpt" ]]; then
        # Format PubMed RCT data
        python scripts/formatting.py data/filtered/rct/dev.jsonl data/formatted_sharegpt/rct/dev.jsonl --format sharegpt --use_prompt_index $PROMPT_INDEX
        python scripts/formatting.py data/filtered/rct/test.jsonl data/formatted_sharegpt/rct/test.jsonl --format sharegpt --use_prompt_index $PROMPT_INDEX
        python scripts/formatting.py data/filtered/rct/train.jsonl data/formatted_sharegpt/rct/train.jsonl --format sharegpt --use_prompt_index $PROMPT_INDEX

        # Format PubMed Non-RCT data
        python scripts/formatting.py data/filtered/non_rct/dev_clean.jsonl data/formatted_sharegpt/non_rct/dev.jsonl --format sharegpt --use_prompt_index $PROMPT_INDEX
        python scripts/formatting.py data/filtered/non_rct/test_clean.jsonl data/formatted_sharegpt/non_rct/test.jsonl --format sharegpt --use_prompt_index $PROMPT_INDEX
        python scripts/formatting.py data/filtered/non_rct/train_clean.jsonl data/formatted_sharegpt/non_rct/train.jsonl --format sharegpt --use_prompt_index $PROMPT_INDEX
    else
        # Format PubMed RCT data
        python scripts/formatting.py data/filtered/rct/dev.jsonl data/formatted_alpaca/rct/dev.jsonl --format alpaca --use_prompt_index $PROMPT_INDEX
        python scripts/formatting.py data/filtered/rct/test.jsonl data/formatted_alpaca/rct/test.jsonl --format alpaca --use_prompt_index $PROMPT_INDEX
        python scripts/formatting.py data/filtered/rct/train.jsonl data/formatted_alpaca/rct/train.jsonl --format alpaca --use_prompt_index $PROMPT_INDEX

        # Format PubMed Non-RCT data
        python scripts/formatting.py data/filtered/non_rct/dev_clean.jsonl data/formatted_alpaca/non_rct/dev.jsonl --format alpaca --use_prompt_index $PROMPT_INDEX
        python scripts/formatting.py data/filtered/non_rct/test_clean.jsonl data/formatted_alpaca/non_rct/test.jsonl --format alpaca --use_prompt_index $PROMPT_INDEX
        python scripts/formatting.py data/filtered/non_rct/train_clean.jsonl data/formatted_alpaca/non_rct/train.jsonl --format alpaca --use_prompt_index $PROMPT_INDEX
    fi
    
    print_success "Data formatting completed"
}

# Function to display summary
display_summary() {
    print_status "Data processing pipeline completed successfully!"
    echo
    echo "Configuration:"
    echo "  Format: $FORMAT_TYPE"
    echo "  Prompt Index: $PROMPT_INDEX"
    echo
    echo "Generated files:"
    if [[ "$SKIP_PROCESSING" == "false" && "$RUN_FORMATTING_ONLY" == "false" ]]; then
        echo "├── data/processed/rct/ (dev.jsonl, test.jsonl, train.jsonl)"
        echo "├── data/processed/non_rct/ (dev_clean.jsonl, test_clean.jsonl, train_clean.jsonl)"
    fi
    if [[ "$SKIP_FILTERING" == "false" && "$RUN_FORMATTING_ONLY" == "false" ]]; then
        echo "├── data/filtered/rct/ (dev.jsonl, test.jsonl, train.jsonl)"
        echo "├── data/filtered/non_rct/ (dev_clean.jsonl, test_clean.jsonl, train_clean.jsonl)"
    fi
    if [[ "$SKIP_FORMATTING" == "false" ]]; then
        if [[ "$FORMAT_TYPE" == "sharegpt" ]]; then
            echo "└── data/formatted_sharegpt/rct/ (dev.jsonl, test.jsonl, train.jsonl)"
            echo "└── data/formatted_sharegpt/non_rct/ (dev.jsonl, test.jsonl, train.jsonl)"
        else
            echo "└── data/formatted_alpaca/rct/ (dev.jsonl, test.jsonl, train.jsonl)"
            echo "└── data/formatted_alpaca/non_rct/ (dev.jsonl, test.jsonl, train.jsonl)"
        fi
    fi
    echo
    print_success "You can now proceed with supervised fine-tuning using the formatted data!"
}

# Main execution
main() {
    echo "=========================================="
    echo "LLM Conclusion Generation - Data Processing"
    echo "=========================================="
    echo
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check if we're in the right directory
    if [[ ! -f "README.md" ]] || [[ ! -d "scripts" ]]; then
        print_error "Please run this script from the LLM_conclusion project root directory"
        exit 1
    fi
    
    # Check if Python is available
    if ! command_exists python; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check virtual environment
    check_venv
    
    # Check data directories
    check_data_directories
    
    # Create output directories
    create_directories
    
    # Execute data processing pipeline
    print_status "Starting data processing pipeline..."
    echo
    
    if [[ "$RUN_FORMATTING_ONLY" == "true" ]]; then
        print_status "Running formatting step only..."
        format_data
    else
        # Step 1: Process raw data
        if [[ "$SKIP_PROCESSING" == "false" ]]; then
            process_pubmed_rct
            process_pubmed_non_rct
            echo
        else
            print_status "Skipping data processing step"
        fi
        
        # Step 2: Filter processed data
        if [[ "$SKIP_FILTERING" == "false" ]]; then
            filter_data
            echo
        else
            print_status "Skipping data filtering step"
        fi
        
        # Step 3: Format data
        if [[ "$SKIP_FORMATTING" == "false" ]]; then
            format_data
            echo
        else
            print_status "Skipping data formatting step"
        fi
    fi
    
    # Display summary
    display_summary
}

# Run main function
main "$@" 