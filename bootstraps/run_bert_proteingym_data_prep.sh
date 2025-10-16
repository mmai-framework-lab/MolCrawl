#!/bin/bash

# BERT ProteinGym data preparation script
# Downloads and preprocesses ProteinGym data for BERT evaluation

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
    echo -e "${BLUE}🧬 BERT ProteinGym Data Preparation${NC}"
    echo "================================================"
    echo -e "${PURPLE}📊 Downloading and preprocessing ProteinGym data for BERT${NC}"
    echo -e "${CYAN}📅 Date: $(date)${NC}"
    echo ""
}

print_config() {
    echo "Configuration:"
    echo "  Output Directory: $OUTPUT_DIR"
    echo "  Max Variants per Assay: $MAX_VARIANTS"
    echo "  Download Data: ${DOWNLOAD:-No}"
    echo "  Sample Only: ${SAMPLE_ONLY:-No}"
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

check_requirements() {
    echo "Checking requirements..."
    
    # Check if output directory is writable
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
    fi
    
    if [ ! -w "$OUTPUT_DIR" ]; then
        echo -e "${RED}❌ Output directory not writable: $OUTPUT_DIR${NC}"
        exit 1
    fi
    
    # Check Python packages
    python -c "import requests, pandas, numpy, zipfile, tqdm" 2>/dev/null || {
        echo -e "${RED}❌ Required Python packages not found${NC}"
        echo "Please ensure requests, pandas, numpy, and tqdm are installed"
        exit 1
    }
    
    echo -e "${GREEN}✅ All requirements satisfied${NC}"
    echo ""
}

run_data_preparation() {
    echo "Running BERT ProteinGym data preparation..."
    echo ""
    
    # Prepare Python command arguments
    PYTHON_ARGS=(
        "bert/proteingym_data_preparation.py"
        "--output_dir" "$OUTPUT_DIR"
        "--max_variants_per_assay" "$MAX_VARIANTS"
    )
    
    # Add download flag if specified
    if [ "$DOWNLOAD" = "true" ]; then
        PYTHON_ARGS+=("--download")
        echo -e "${CYAN}📥 Downloading ProteinGym data from official source${NC}"
    fi
    
    # Add sample only flag if specified
    if [ "$SAMPLE_ONLY" = "true" ]; then
        PYTHON_ARGS+=("--sample_only")
        echo -e "${CYAN}📝 Creating sample dataset only${NC}"
    fi
    
    # Run the preparation
    echo -e "${GREEN}🚀 Starting data preparation...${NC}"
    python "${PYTHON_ARGS[@]}"
    
    PREP_EXIT_CODE=$?
    if [ $PREP_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}❌ Data preparation failed with exit code $PREP_EXIT_CODE${NC}"
        exit $PREP_EXIT_CODE
    fi
}

analyze_results() {
    echo ""
    echo "BERT ProteinGym data preparation completed!"
    echo "=== Preparation Results Summary ==="
    
    # Check output files
    echo "Output files generated:"
    ls -la "$OUTPUT_DIR"
    
    # Show statistics if available
    if [ -f "$OUTPUT_DIR/bert_proteingym_statistics.txt" ]; then
        echo ""
        echo "Dataset Statistics:"
        head -20 "$OUTPUT_DIR/bert_proteingym_statistics.txt"
    fi
    
    # Show metadata if available
    if [ -f "$OUTPUT_DIR/bert_proteingym_dataset.json" ]; then
        echo ""
        echo "Dataset Metadata:"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/bert_proteingym_dataset.json', 'r') as f:
        data = json.load(f)
        metadata = data.get('metadata', {})
        print(f'  Total variants: {metadata.get(\"total_variants\", \"N/A\")}')
        print(f'  Unique assays: {metadata.get(\"unique_assays\", \"N/A\")}')
        print(f'  DMS score range: {metadata.get(\"dms_score_range\", \"N/A\")}')
        print(f'  Avg sequence length: {metadata.get(\"avg_sequence_length\", \"N/A\"):.1f}')
        print(f'  Processing date: {metadata.get(\"processing_date\", \"N/A\")}')
except Exception as e:
    print(f'Could not read metadata: {e}')
"
    fi
}

print_completion_message() {
    echo ""
    echo -e "${GREEN}🎉 BERT ProteinGym Data Preparation Completed Successfully!${NC}"
    echo "================================================================="
    echo -e "${CYAN}📁 Output: $OUTPUT_DIR${NC}"
    echo -e "${CYAN}📋 Logs: logs/${NC}"
    echo -e "${CYAN}🧬 Ready for BERT ProteinGym evaluation${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Run BERT ProteinGym evaluation:"
    echo "   ./run_bert_proteingym_evaluation.sh --dataset $OUTPUT_DIR/bert_proteingym_dataset.csv"
    echo "2. Or test with sample data:"
    echo "   ./run_bert_proteingym_evaluation.sh --dataset $OUTPUT_DIR/bert_proteingym_sample.csv"
    echo ""
}

cleanup() {
    echo "Cleaning up temporary files..."
    # Add any cleanup operations here
    echo "Cleanup completed."
}

# Main execution
main() {
    # Configuration defaults
    OUTPUT_DIR="${OUTPUT_DIR:-./bert_proteingym_data}"
    MAX_VARIANTS="${MAX_VARIANTS:-1000}"
    DOWNLOAD="${DOWNLOAD:-false}"
    SAMPLE_ONLY="${SAMPLE_ONLY:-false}"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output_dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --max_variants)
                MAX_VARIANTS="$2"
                shift 2
                ;;
            --download)
                DOWNLOAD=true
                shift
                ;;
            --sample_only)
                SAMPLE_ONLY=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --output_dir PATH          Output directory for processed data"
                echo "                             (default: ./bert_proteingym_data)"
                echo "  --max_variants N           Maximum variants per assay (default: 1000)"
                echo "  --download                 Download ProteinGym data from official source"
                echo "  --sample_only              Create sample dataset only"
                echo "  --help                     Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Create output and log directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "logs"
    
    # Print header and configuration
    print_header
    print_config
    
    # Activate conda environment
    activate_conda_env
    
    echo -e "${GREEN}🚀 Starting BERT ProteinGym Data Preparation Pipeline...${NC}"
    echo ""
    
    # Main preparation steps
    check_requirements
    run_data_preparation
    analyze_results
    print_completion_message
    cleanup
}

# Trap to handle script interruption
trap cleanup EXIT

# Run main function with all arguments
main "$@"