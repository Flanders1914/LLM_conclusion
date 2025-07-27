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
    
    print_success "Data directories check passed"
}

# Function to create output directories
create_directories() {
    print_status "Creating output directories..."
    
    mkdir -p data/processed/rct
    mkdir -p data/processed/non_rct
    mkdir -p data/filtered/rct
    mkdir -p data/filtered/non_rct
    mkdir -p data/formatted_sharegpt/rct
    mkdir -p data/formatted_sharegpt/non_rct
    
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

# Function to format data into ShareGPT format
format_data() {
    print_status "Formatting data into ShareGPT format..."
    
    # Format PubMed RCT data
    python scripts/formatting.py data/filtered/rct/dev.jsonl data/formatted_sharegpt/rct/dev.jsonl --format sharegpt
    python scripts/formatting.py data/filtered/rct/test.jsonl data/formatted_sharegpt/rct/test.jsonl --format sharegpt
    python scripts/formatting.py data/filtered/rct/train.jsonl data/formatted_sharegpt/rct/train.jsonl --format sharegpt

    
    # Format PubMed Non-RCT data
    python scripts/formatting.py data/filtered/non_rct/dev_clean.jsonl data/formatted_sharegpt/non_rct/dev.jsonl --format sharegpt
    python scripts/formatting.py data/filtered/non_rct/test_clean.jsonl data/formatted_sharegpt/non_rct/test.jsonl --format sharegpt
    python scripts/formatting.py data/filtered/non_rct/train_clean.jsonl data/formatted_sharegpt/non_rct/train.jsonl --format sharegpt
    
    print_success "Data formatting completed"
}

# Function to display summary
display_summary() {
    print_status "Data processing pipeline completed successfully!"
    echo
    echo "Generated files:"
    echo "├── data/processed/rct/ (dev.jsonl, test.jsonl, train.jsonl)"
    echo "├── data/processed/non_rct/ (dev_clean.jsonl, test_clean.jsonl, train_clean.jsonl)"
    echo "├── data/filtered/rct/ (dev.jsonl, test.jsonl, train.jsonl)"
    echo "├── data/filtered/non_rct/ (dev_clean.jsonl, test_clean.jsonl, train_clean.jsonl)"
    echo "└── data/formatted_sharegpt/rct/ and data/formatted_sharegpt/non_rct/ (ShareGPT format files)"
    echo
    print_success "You can now proceed with supervised fine-tuning using the formatted data!"
}

# Main execution
main() {
    echo "=========================================="
    echo "LLM Conclusion Generation - Data Processing"
    echo "=========================================="
    echo
    
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
    
    # Step 1: Process raw data
    process_pubmed_rct
    process_pubmed_non_rct
    echo
    
    # Step 2: Filter processed data
    filter_data
    echo
    
    # Step 3: Format data
    format_data
    echo
    
    # Display summary
    display_summary
}

# Run main function
main "$@" 