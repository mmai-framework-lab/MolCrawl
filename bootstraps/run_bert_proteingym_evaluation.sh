#!/bin/bash

# BERT ProteinGym evaluation script for protein_sequence model
# Based on the trained BERT model for protein sequences

set -e  # Exit on any error

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function definitions
print_header() {
    echo -e "${BLUE}🧬 Independent BERT Protein Sequence - ProteinGym Evaluation${NC}"
    echo "================================================================"
    echo -e "${PURPLE}🤖 BERT-based protein fitness prediction (Independent Implementation)${NC}"
    echo -e "${CYAN}📅 Date: $(date)${NC}"
    echo -e "${GREEN}🚀 Using trained BERT model with safetensors${NC}"
    echo ""
}

print_config() {
    echo "Configuration:"
    echo "  Model Path: $MODEL_PATH"
    if [ "$TOKENIZER_PATH" = "None" ] || [ -z "$TOKENIZER_PATH" ]; then
        echo "  Tokenizer: EsmSequenceTokenizer (built-in)"
    else
        echo "  Tokenizer Path: $TOKENIZER_PATH"
    fi
    echo "  Dataset Path: $DATASET_PATH"
    echo "  Output Directory: $OUTPUT_DIR"
    echo "  Sample Size: ${SAMPLE_SIZE:-All variants}"
    echo ""
}

check_requirements() {
    echo "Checking requirements..."
    
    # Check if model exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}❌ Model directory not found: $MODEL_PATH${NC}"
        exit 1
    fi
    
    # Check for model file (safetensors or pytorch)
    if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
        echo -e "${RED}❌ No model file found in $MODEL_PATH${NC}"
        echo "Expected: model.safetensors or pytorch_model.bin"
        exit 1
    fi
    
    # Check tokenizer (protein_sequence uses EsmSequenceTokenizer, not SentencePiece)
    if [ "$TOKENIZER_PATH" != "None" ] && [ ! -z "$TOKENIZER_PATH" ] && [ ! -f "$TOKENIZER_PATH" ]; then
        echo -e "${YELLOW}⚠️  SentencePiece tokenizer not found: $TOKENIZER_PATH${NC}"
        echo -e "${CYAN}ℹ️  Will use EsmSequenceTokenizer for protein_sequence${NC}"
    fi
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo -e "${RED}❌ Dataset not found: $DATASET_PATH${NC}"
        echo "Please specify a valid ProteinGym dataset path"
        exit 1
    fi
    
    # Check dataset format and content
    if [ -f "$DATASET_PATH" ]; then
        case "$DATASET_PATH" in
            *.csv)
                VARIANT_COUNT=$(tail -n +2 "$DATASET_PATH" | wc -l)
                echo -e "${GREEN}📊 Dataset contains $VARIANT_COUNT variants${NC}"
                ;;
            *.tsv)
                VARIANT_COUNT=$(tail -n +2 "$DATASET_PATH" | wc -l)
                echo -e "${GREEN}📊 Dataset contains $VARIANT_COUNT variants${NC}"
                ;;
            *.json)
                echo -e "${GREEN}📊 JSON dataset detected${NC}"
                ;;
            *)
                echo -e "${YELLOW}⚠️  Unknown dataset format${NC}"
                ;;
        esac
    fi
    
    echo "All requirements satisfied."
    echo ""
}

activate_conda_env() {
    # Activate conda environment
    if [ -f "./miniconda/etc/profile.d/conda.sh" ]; then
        source ./miniconda/etc/profile.d/conda.sh
        conda activate conda
        echo -e "${GREEN}✅ Conda environment activated${NC}"
    else
        echo -e "${YELLOW}⚠️  Conda environment not found, using system Python${NC}"
    fi
    
    # Set environment variables for protein_sequence
    export LEARNING_SOURCE_DIR='learning_source_202508'
    echo -e "${CYAN}🌍 Environment variables set for protein_sequence${NC}"
}

run_evaluation() {
    echo "Running BERT ProteinGym evaluation..."
    echo "This may take several minutes depending on dataset size and model complexity."
    echo ""
    
    # Check CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
        echo "Using GPU for evaluation"
    else
        echo "CUDA not available, using CPU"
    fi
    echo ""
    
    # Prepare Python command arguments
    local PYTHON_ARGS=(
        "bert/proteingym_evaluation.py"
        "--model_path" "$MODEL_PATH"
        "--proteingym_data" "$DATASET_PATH"
        "--device" "${DEVICE:-cuda}"
        "--batch_size" "${BATCH_SIZE:-16}"
    )
    
    # Add tokenizer path only if it's not None
    if [ "$TOKENIZER_PATH" != "None" ] && [ ! -z "$TOKENIZER_PATH" ]; then
        PYTHON_ARGS+=("--tokenizer_path" "$TOKENIZER_PATH")
    fi
    
    # Add sample size if specified
    if [ -n "$SAMPLE_SIZE" ]; then
        PYTHON_ARGS+=("--sample_size" "$SAMPLE_SIZE")
    fi
    
    # Run the evaluation
    python "${PYTHON_ARGS[@]}"
    
    EVAL_EXIT_CODE=$?
    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}❌ Evaluation failed with exit code $EVAL_EXIT_CODE${NC}"
        exit $EVAL_EXIT_CODE
    fi
}

analyze_bert_results() {
    echo ""
    echo "BERT ProteinGym evaluation completed!"
    echo "=== Evaluation Completed ==="
    echo "Results have been automatically saved to the structured output directory."
    echo "Check the Python execution log above for the exact output directory path."
    
    # Results will be in: ${LEARNING_SOURCE_DIR}/protein_sequence/report/bert_proteingym_*_YYYYMMDD_HHMMSS/
    echo "Expected location: \${LEARNING_SOURCE_DIR}/protein_sequence/report/bert_proteingym_*/
    
    # BERT-specific analysis
    echo -e "${BLUE}🔍 BERT-Specific Analysis${NC}"
    echo "========================="
    
    if [ -f "$OUTPUT_DIR/bert_proteingym_results.json" ]; then
        python -c "
import json
with open('$OUTPUT_DIR/bert_proteingym_results.json', 'r') as f:
    results = json.load(f)

print('🧠 BERT Model Insights:')
spearman = results['spearman_correlation']
print(f'🎯 Performance Summary:')
print(f'   Spearman correlation: {spearman:.3f}')
print(f'   Pearson correlation: {results[\"pearson_correlation\"]:.3f}')
print(f'   MAE: {results[\"mae\"]:.3f}')

print(f'')
print(f'🧬 Fitness Analysis:')
print(f'   True Score Range: {results[\"true_score_stats\"][\"min\"]:.3f} to {results[\"true_score_stats\"][\"max\"]:.3f}')
print(f'   Predicted Range: {results[\"predicted_score_stats\"][\"min\"]:.3f} to {results[\"predicted_score_stats\"][\"max\"]:.3f}')

print(f'')
print(f'📊 Model Characteristics:')
print(f'   • Bidirectional context analysis')
print(f'   • Masked language modeling approach')
print(f'   • Sequence representation learning')
print(f'   • Independent of generative models')

print(f'')
# Performance assessment
if spearman > 0.7:
    print(f'✅ Excellent protein fitness prediction performance')
elif spearman > 0.5:
    print(f'🟢 Good protein fitness prediction performance')
elif spearman > 0.3:
    print(f'🟡 Moderate protein fitness prediction performance')
else:
    print(f'⚠️  Limited protein fitness prediction performance')
"
    fi
}

print_completion_message() {
    echo ""
    echo -e "${GREEN}🎉 Independent BERT ProteinGym Evaluation Completed Successfully!${NC}"
    echo "=============================================================="
    echo -e "${CYAN}📁 Results saved to structured directory under \${LEARNING_SOURCE_DIR}/protein_sequence/report/${NC}"
    echo -e "${CYAN}📋 Logs: logs/${NC}"
    echo -e "${CYAN}🧬 Model: Trained BERT Protein Sequence Model${NC}"
    echo -e "${CYAN}📊 Method: Independent fitness assessment${NC}"
    echo ""
}

cleanup() {
    echo "Cleaning up temporary files..."
    # Add any cleanup operations here
    echo "Cleanup completed."
}

# Main execution
main() {
    # Set LEARNING_SOURCE_DIR if not already set
    export LEARNING_SOURCE_DIR="${LEARNING_SOURCE_DIR:-$(dirname "$0")/learning_source_202508}"
    
    # Configuration
    MODEL_PATH="${MODEL_PATH:-runs_train_bert_protein_sequence/checkpoint-2000}"
    TOKENIZER_PATH="${TOKENIZER_PATH:-None}"  # protein_sequenceはEsmSequenceTokenizerを使用
    OUTPUT_DIR=""  # Will be auto-generated by the Python script
    DEVICE="${DEVICE:-cuda}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model_path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --tokenizer_path)
                TOKENIZER_PATH="$2"
                shift 2
                ;;
            --dataset)
                DATASET_PATH="$2"
                shift 2
                ;;
                        --tokenizer_path)
            --sample_size)
                SAMPLE_SIZE="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --batch_size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --create_sample_data)
                CREATE_SAMPLE_DATA=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --model_path PATH          Path to trained BERT model"
                echo "                             (default: runs_train_bert_protein_sequence/checkpoint-2000)"
                echo "  --tokenizer_path PATH      Path to tokenizer"
                echo "                             (default: EsmSequenceTokenizer built-in)"
                echo "  --dataset PATH             Path to ProteinGym dataset (required)"
                echo "  --output_dir PATH          Output directory"
                echo "                             (default: ./bert_proteingym_evaluation_results)"
                echo "  --sample_size N            Number of variants to evaluate (default: all)"
                echo "  --device DEVICE            Device to use (default: cuda)"
                echo "  --batch_size N             Batch size (default: 16)"
                echo "  --create_sample_data       Create sample dataset for testing"
                echo "  --help                     Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check if dataset path is provided
    if [ -z "$DATASET_PATH" ] && [ "$CREATE_SAMPLE_DATA" != "true" ]; then
        echo -e "${RED}❌ Dataset path is required${NC}"
        echo "Use --dataset PATH to specify ProteinGym data file"
        echo "Use --create_sample_data to create test data"
        exit 1
    fi
    
    # Output directory will be auto-created by the Python script
    mkdir -p "logs"
    
    # Print header and configuration
    print_header
    
    # Handle sample data creation
    if [ "$CREATE_SAMPLE_DATA" = "true" ]; then
        echo "Creating sample ProteinGym data..."
        SAMPLE_DATA_PATH="${DATASET_PATH:-./sample_proteingym_data.csv}"
        SAMPLE_ARGS=("bert/proteingym_evaluation.py" "--create_sample_data" "--proteingym_data" "$SAMPLE_DATA_PATH" "--model_path" "$MODEL_PATH")
        if [ "$TOKENIZER_PATH" != "None" ] && [ ! -z "$TOKENIZER_PATH" ]; then
            SAMPLE_ARGS+=("--tokenizer_path" "$TOKENIZER_PATH")
        fi
        python "${SAMPLE_ARGS[@]}"
        echo -e "${GREEN}✅ Sample data created at: $SAMPLE_DATA_PATH${NC}"
        echo "Run again with --dataset $SAMPLE_DATA_PATH to evaluate"
        exit 0
    fi
    
    print_config
    
    # Activate conda environment
    activate_conda_env
    
    echo -e "${GREEN}🚀 Starting Independent BERT ProteinGym Evaluation Pipeline...${NC}"
    echo ""
    
    # Main evaluation steps
    check_requirements
    run_evaluation
    analyze_bert_results
    print_completion_message
    cleanup
}

# Trap to handle script interruption
trap cleanup EXIT

# Run main function with all arguments
main "$@"