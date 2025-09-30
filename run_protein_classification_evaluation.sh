#!/bin/bash

# Protein Sequence GPT2 Classification Evaluation Script
# This script evaluates a trained protein sequence GPT2 model using classification metrics

set -e

# Default values
MODEL_PATH=""
DATA_PATH=""
OUTPUT_DIR="./protein_classification_results"
CREATE_SAMPLE="false"
THRESHOLD="0.0"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Evaluate protein sequence GPT2 model with classification metrics

OPTIONS:
    -m, --model_path PATH       Path to trained GPT2 model checkpoint (required)
    -d, --data_path PATH        Path to evaluation dataset CSV
    -o, --output_dir PATH       Output directory for results (default: ./protein_classification_results)
    -s, --create_sample         Create sample evaluation dataset
    -t, --threshold FLOAT       Threshold for binary classification (default: 0.0)
    -h, --help                  Show this help message

EXAMPLES:
    # Evaluate with sample data
    $0 -m gpt2-output/protein_sequence-small/ckpt.pt -s
    
    # Evaluate with custom dataset
    $0 -m gpt2-output/protein_sequence-small/ckpt.pt -d my_protein_variants.csv
    
    # Specify custom output directory
    $0 -m gpt2-output/protein_sequence-small/ckpt.pt -s -o ./my_results

REQUIRED CSV FORMAT:
    The evaluation dataset should contain the following columns:
    - sequence: Protein amino acid sequence
    - variant_pos: Position of variant (0-indexed)
    - ref_aa: Reference amino acid
    - alt_aa: Alternative amino acid
    - pathogenic: Binary label (1=pathogenic, 0=benign)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--create_sample)
            CREATE_SAMPLE="true"
            shift
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Model path is required (-m/--model_path)"
    usage
    exit 1
fi

# Check if model file exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

# Setup environment
echo "╔════════════════════════════════════════════════════════════╗"
echo "║      Protein Sequence GPT2 Classification Evaluation      ║"
echo "║                  Binary Classification Metrics             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

echo "[INFO] Setting up evaluation environment..."

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Create conda environment activation command
CONDA_ENV_NAME="bert_protein"  # Assuming same env as BERT

# Check if conda environment exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "[INFO] Activating conda environment: $CONDA_ENV_NAME"
    CONDA_ACTIVATE="conda activate $CONDA_ENV_NAME"
else
    echo "[WARN] Conda environment '$CONDA_ENV_NAME' not found"
    echo "[INFO] Available environments:"
    conda env list
    echo "[WARN] Using base conda environment"
    CONDA_ACTIVATE=""
fi

# Check Python dependencies
echo "[INFO] Checking Python dependencies..."
eval $CONDA_ACTIVATE

python -c "
import sys
required_packages = ['torch', 'sklearn', 'pandas', 'numpy']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Please install them using: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('All required packages are available')
"

if [[ $? -ne 0 ]]; then
    echo "[ERROR] Missing required Python packages"
    exit 1
fi

# Build command arguments
EVAL_ARGS="--model_path \"$MODEL_PATH\""
EVAL_ARGS="$EVAL_ARGS --output_dir \"$OUTPUT_DIR\""
EVAL_ARGS="$EVAL_ARGS --threshold $THRESHOLD"

if [[ "$CREATE_SAMPLE" == "true" ]]; then
    EVAL_ARGS="$EVAL_ARGS --create_sample_data"
fi

if [[ -n "$DATA_PATH" ]]; then
    if [[ ! -f "$DATA_PATH" ]]; then
        echo "[ERROR] Evaluation dataset not found: $DATA_PATH"
        exit 1
    fi
    EVAL_ARGS="$EVAL_ARGS --data_path \"$DATA_PATH\""
fi

# Display configuration
echo
echo "📋 Evaluation Configuration:"
echo "============================"
echo "Model Path:        $MODEL_PATH"
echo "Data Path:         ${DATA_PATH:-"Will create sample data"}"
echo "Output Directory:  $OUTPUT_DIR"
echo "Create Sample:     $CREATE_SAMPLE"
echo "Threshold:         $THRESHOLD"
echo "Conda Environment: ${CONDA_ENV_NAME:-"base"}"
echo

# Run evaluation
echo "[INFO] 🚀 Starting protein sequence GPT2 classification evaluation..."

EVAL_COMMAND="python bert/protein_classification_evaluation.py $EVAL_ARGS"
echo "Command: $EVAL_COMMAND"
echo

if [[ -n "$CONDA_ACTIVATE" ]]; then
    eval $CONDA_ACTIVATE && eval $EVAL_COMMAND
else
    eval $EVAL_COMMAND
fi

# Check if evaluation was successful
if [[ $? -eq 0 ]]; then
    echo
    echo "[INFO] 🎉 Evaluation completed successfully!"
    echo
    echo "📊 Generated Files:"
    echo "==================="
    find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.csv" -o -name "*.png" | while read file; do
        echo "  📈 $(basename "$file")"
    done
    echo
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo
    echo "📋 Evaluation Metrics:"
    echo "======================"
    if [[ -f "$OUTPUT_DIR/protein_classification_results.json" ]]; then
        python -c "
import json
with open('$OUTPUT_DIR/protein_classification_results.json', 'r') as f:
    results = json.load(f)
    metrics = results['metrics']
    print(f'  • Accuracy:    {metrics[\"Accuracy\"]:.4f}')
    print(f'  • Precision:   {metrics[\"Precision\"]:.4f}')
    print(f'  • Recall:      {metrics[\"Recall\"]:.4f}')
    print(f'  • F1-score:    {metrics[\"F1-score\"]:.4f}')
    print(f'  • ROC-AUC:     {metrics[\"ROC-AUC\"]:.4f}')
    print(f'  • PR-AUC:      {metrics[\"PR-AUC\"]:.4f}')
    print(f'  • Sensitivity: {metrics[\"Sensitivity\"]:.4f}')
    print(f'  • Specificity: {metrics[\"Specificity\"]:.4f}')
"
    fi
    echo
    
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                      🎉 SUCCESS! 🎉                      ║"
    echo "║     Protein Classification Evaluation Complete             ║"
    echo "╚════════════════════════════════════════════════════════════╝"
else
    echo "[ERROR] ❌ Evaluation failed"
    exit 1
fi