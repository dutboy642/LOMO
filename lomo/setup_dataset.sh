# #!/bin/bash

# set -e  # Exit on any error

# echo "🚀 LOMO Dataset Setup Script"
# echo "============================="

# # Colors for output
# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m' # No Color

# # Function to print colored output
# print_status() {
#     echo -e "${GREEN}[INFO]${NC} $1"
# }

# print_warning() {
#     echo -e "${YELLOW}[WARN]${NC} $1"
# }

# print_error() {
#     echo -e "${RED}[ERROR]${NC} $1"
# }

# print_header() {
#     echo -e "${BLUE}=== $1 ===${NC}"
# }

# # Default values
# DATASET_TYPE=""
# LANGUAGE="ja"
# CUSTOM_PATH=""
# FORCE_DOWNLOAD=false

# # Parse command line arguments
# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --dataset|-d)
#             DATASET_TYPE="$2"
#             shift 2
#             ;;
#         --language|-l)
#             LANGUAGE="$2"
#             shift 2
#             ;;
#         --path|-p)
#             CUSTOM_PATH="$2"
#             shift 2
#             ;;
#         --force|-f)
#             FORCE_DOWNLOAD=true
#             shift
#             ;;
#         --help|-h)
#             echo "Usage: $0 [OPTIONS]"
#             echo ""
#             echo "Options:"
#             echo "  -d, --dataset TYPE    Dataset type: wikipedia, custom_text, custom_json"
#             echo "  -l, --language LANG   Language for Wikipedia (ja, vi, en, etc.)"
#             echo "  -p, --path PATH       Path to custom dataset file"
#             echo "  -f, --force          Force re-download even if exists"
#             echo "  -h, --help           Show this help message"
#             echo ""
#             echo "Examples:"
#             echo "  $0 --dataset wikipedia --language ja"
#             echo "  $0 --dataset custom_text --path /path/to/your/corpus.txt"
#             exit 0
#             ;;
#         *)
#             print_error "Unknown option: $1"
#             echo "Use --help for usage information"
#             exit 1
#             ;;
#     esac
# done

# # Interactive mode if no dataset specified
# if [[ -z "$DATASET_TYPE" ]]; then
#     print_header "Interactive Dataset Setup"
#     echo "Please choose a dataset type:"
#     echo "1. Wikipedia dataset (auto-download)"
#     echo "2. Custom text file"
#     echo "3. Custom JSON/JSONL file"
#     echo ""
#     read -p "Enter your choice (1-3): " choice
    
#     case $choice in
#         1)
#             DATASET_TYPE="wikipedia"
#             echo "Available languages: ja (Japanese), vi (Vietnamese), en (English), ko (Korean), zh (Chinese)"
#             read -p "Enter language code [default: ja]: " lang_input
#             LANGUAGE=${lang_input:-$LANGUAGE}
#             ;;
#         2)
#             DATASET_TYPE="custom_text"
#             read -p "Enter path to your text file: " CUSTOM_PATH
#             ;;
#         3)
#             DATASET_TYPE="custom_json"
#             read -p "Enter path to your JSON/JSONL file: " CUSTOM_PATH
#             ;;
#         *)
#             print_error "Invalid choice"
#             exit 1
#             ;;
#     esac
# fi

# print_header "Setup Configuration"
# echo "Dataset Type: $DATASET_TYPE"
# if [[ "$DATASET_TYPE" == "wikipedia" ]]; then
#     echo "Language: $LANGUAGE"
# elif [[ ! -z "$CUSTOM_PATH" ]]; then
#     echo "Custom Path: $CUSTOM_PATH"
# fi
# echo "Force Download: $FORCE_DOWNLOAD"
# echo ""

# # Create data directory
# mkdir -p data
# print_status "Created data directory"

# # Process based on dataset type
# case $DATASET_TYPE in
#     wikipedia)
#         print_header "Setting up Wikipedia Dataset"
        
#         # Check if Python script exists
#         if [[ ! -f "src/download_wiki_dataset.py" ]]; then
#             print_error "download_wiki_dataset.py not found!"
#             exit 1
#         fi
        
#         # Download Wikipedia dataset
#         print_status "Downloading Wikipedia dataset for language: $LANGUAGE"
#         if [[ "$FORCE_DOWNLOAD" == true ]]; then
#             python src/download_wiki_dataset.py --language "$LANGUAGE" --force
#         else
#             python src/download_wiki_dataset.py --language "$LANGUAGE"
#         fi
        
#         # Update config file
#         CONFIG_FILE="config/args_continued_pretraining.yaml"
#         if [[ -f "$CONFIG_FILE" ]]; then
#             print_status "Updating configuration file: $CONFIG_FILE"
#             sed -i "s/dataset_name: .*/dataset_name: 'wikipedia_${LANGUAGE}'/" "$CONFIG_FILE"
#             print_status "Configuration updated with dataset_name: 'wikipedia_${LANGUAGE}'"
#         fi
#         ;;
        
#     custom_text)
#         print_header "Setting up Custom Text Dataset"
        
#         if [[ -z "$CUSTOM_PATH" ]]; then
#             print_error "Custom path not specified!"
#             exit 1
#         fi
        
#         if [[ ! -f "$CUSTOM_PATH" ]]; then
#             print_error "File not found: $CUSTOM_PATH"
#             exit 1
#         fi
        
#         # Copy to data directory
#         cp "$CUSTOM_PATH" data/custom_text.txt
#         print_status "Copied $CUSTOM_PATH to data/custom_text.txt"
        
#         # Update config
#         CONFIG_FILE="config/args_continued_pretraining.yaml"
#         if [[ -f "$CONFIG_FILE" ]]; then
#             sed -i "s/dataset_name: .*/dataset_name: 'custom_text'/" "$CONFIG_FILE"
#             print_status "Configuration updated with dataset_name: 'custom_text'"
#         fi
#         ;;
        
#     custom_json)
#         print_header "Setting up Custom JSON Dataset"
        
#         if [[ -z "$CUSTOM_PATH" ]]; then
#             print_error "Custom path not specified!"
#             exit 1
#         fi
        
#         if [[ ! -f "$CUSTOM_PATH" ]]; then
#             print_error "File not found: $CUSTOM_PATH"
#             exit 1
#         fi
        
#         # Determine file extension and copy accordingly
#         if [[ "$CUSTOM_PATH" == *.jsonl ]]; then
#             cp "$CUSTOM_PATH" data/custom_data.jsonl
#             dataset_name="custom_jsonl"
#             print_status "Copied $CUSTOM_PATH to data/custom_data.jsonl"
#         else
#             cp "$CUSTOM_PATH" data/custom_data.json
#             dataset_name="custom_jsonl"
#             print_status "Copied $CUSTOM_PATH to data/custom_data.json"
#         fi
        
#         # Update config
#         CONFIG_FILE="config/args_continued_pretraining.yaml"
#         if [[ -f "$CONFIG_FILE" ]]; then
#             sed -i "s/dataset_name: .*/dataset_name: '$dataset_name'/" "$CONFIG_FILE"
#             print_status "Configuration updated with dataset_name: '$dataset_name'"
#         fi
#         ;;
        
#     *)
#         print_error "Unknown dataset type: $DATASET_TYPE"
#         exit 1
#         ;;
# esac

# print_header "Setup Complete!"
# echo ""
# print_status "Dataset setup completed successfully!"
# print_status "You can now run training with:"
# echo ""
# echo "  CUDA_VISIBLE_DEVICES=0 bash run_continued_pretraining.sh"
# echo ""
# print_warning "Make sure to review and adjust the configuration in:"
# echo "  config/args_continued_pretraining.yaml"
# echo ""
# print_status "Happy training! 🎉"