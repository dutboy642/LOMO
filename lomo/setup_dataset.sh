#!/bin/bash

set -e  # Exit on any error

echo "🚀 LOMO Dataset Setup Script (YAML-based Config)"
echo "==============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Default config file
CONFIG_FILE="config/args_continued_pretraining.yaml"
DATASET_PATH=""
TEXT_COLUMN="text"
SAMPLE_SIZE="1000"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dataset-path|-d)
            DATASET_PATH="$2"
            shift 2
            ;;
        --text-column|-t)
            TEXT_COLUMN="$2"
            shift 2
            ;;
        --sample-size|-s)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --config FILE         Config file to update [default: config/args_continued_pretraining.yaml]"
            echo "  -d, --dataset-path PATH   Dataset path (local file or HF repo)"
            echo "  -t, --text-column COL     Text column name [default: text]"
            echo "  -s, --sample-size SIZE    Number of samples (-1 for all) [default: 1000]"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Local JSON file"
            echo "  $0 -d 'data/my_dataset.json' -t 'content' -s 5000"
            echo ""
            echo "  # HuggingFace dataset"
            echo "  $0 -d 'username/dataset-repo' -t 'text' -s -1"
            echo ""
            echo "  # Local text file"
            echo "  $0 -d 'data/corpus.txt' -s 10000"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Interactive mode if no dataset path specified
if [[ -z "$DATASET_PATH" ]]; then
    print_header "Interactive Dataset Configuration"
    echo "Choose dataset type:"
    echo "1. Local JSON/JSONL file"
    echo "2. Local text file (.txt)"  
    echo "3. HuggingFace dataset repository"
    echo ""
    read -p "Enter your choice (1-3): " choice
    
    case $choice in
        1)
            read -p "Enter path to JSON/JSONL file: " DATASET_PATH
            read -p "Enter text column name [default: text]: " col_input
            TEXT_COLUMN=${col_input:-$TEXT_COLUMN}
            ;;
        2)
            read -p "Enter path to text file: " DATASET_PATH
            TEXT_COLUMN="text"  # Auto-assigned for .txt files
            ;;
        3)
            read -p "Enter HuggingFace repo (username/repo-name): " DATASET_PATH
            read -p "Enter text column name [default: text]: " col_input
            TEXT_COLUMN=${col_input:-$TEXT_COLUMN}
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
    
    read -p "Enter sample size (-1 for all data) [default: 1000]: " size_input
    SAMPLE_SIZE=${size_input:-$SAMPLE_SIZE}
fi

print_header "Configuration Summary"
echo "Config File: $CONFIG_FILE"
echo "Dataset Path: $DATASET_PATH"
echo "Text Column: $TEXT_COLUMN"
echo "Sample Size: $SAMPLE_SIZE"
echo ""

# Validate inputs
if [[ -z "$DATASET_PATH" ]]; then
    print_error "Dataset path is required!"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Update configuration file
print_status "Updating configuration file..."

# Use sed to update YAML config
sed -i "s|dataset_path: .*|dataset_path: '$DATASET_PATH'|" "$CONFIG_FILE"
sed -i "s|text_column: .*|text_column: '$TEXT_COLUMN'|" "$CONFIG_FILE"
sed -i "s|sample_size: .*|sample_size: $SAMPLE_SIZE|" "$CONFIG_FILE"

print_status "Configuration updated successfully!"
print_status "Updated fields:"
echo "  dataset_path: '$DATASET_PATH'"
echo "  text_column: '$TEXT_COLUMN'" 
echo "  sample_size: $SAMPLE_SIZE"

print_header "Setup Complete!"
print_status "Dataset configuration updated in: $CONFIG_FILE"
print_status "You can now run training with:"
echo ""
echo "  python src/train_lomo_continued_pretraining.py $CONFIG_FILE"
echo ""
print_warning "Review your configuration before training:"
echo "  cat $CONFIG_FILE"
echo ""
print_status "Happy training! 🎉"