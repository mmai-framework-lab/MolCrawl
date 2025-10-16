#!/bin/bash

#==============================================================================
# BERT ProteinGym Evaluation Results Visualization Script
#==============================================================================
# This script generates comprehensive visualizations and analysis plots
# for BERT ProteinGym evaluation results
#
# Usage: ./run_bert_visualization.sh [OPTIONS]
# 
# Author: Generated for BERT ProteinGym evaluation
# Date: $(date +%Y-%m-%d)
#==============================================================================

set -euo pipefail

# Default values
RESULTS_DIR=""
OUTPUT_DIR=""
CONDA_ENV_NAME="bert_protein"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#==============================================================================
# Helper Functions
#==============================================================================

print_header() {
    echo -e "${PURPLE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║           BERT ProteinGym Results Visualization           ║"
    echo "║                  Comprehensive Analysis                    ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    cat << EOF
${CYAN}Usage: $0 [OPTIONS]${NC}

${YELLOW}Required Options:${NC}
    --results_dir DIR       Directory containing BERT evaluation results
                           (should contain bert_proteingym_results.json and 
                            bert_proteingym_detailed_results.csv)

${YELLOW}Optional Options:${NC}
    --output_dir DIR        Output directory for visualizations 
                           (default: results_dir/plots)
    --conda_env NAME        Conda environment name (default: bert_protein)
    --help, -h              Show this help message

${YELLOW}Examples:${NC}
    # Basic usage
    $0 --results_dir ./bert_proteingym_evaluation_results

    # With custom output directory
    $0 --results_dir ./results --output_dir ./my_plots

    # Using specific conda environment
    $0 --results_dir ./results --conda_env my_env

${YELLOW}Generated Outputs:${NC}
    📊 bert_correlation_analysis.png     - Correlation and residual plots
    📈 bert_distribution_analysis.png    - Score distribution analysis
    🎯 bert_performance_metrics.png      - Performance metrics dashboard
    📝 bert_evaluation_report.md         - Comprehensive analysis report

EOF
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_conda_env() {
    local env_name=$1
    
    if ! conda info --envs | grep -q "^${env_name}"; then
        log_warn "Conda environment '${env_name}' not found"
        log_info "Available environments:"
        conda info --envs
        return 1
    fi
    return 0
}

validate_results_dir() {
    local results_dir=$1
    
    if [[ ! -d "$results_dir" ]]; then
        log_error "Results directory does not exist: $results_dir"
        return 1
    fi
    
    local json_file="${results_dir}/bert_proteingym_results.json"
    local csv_file="${results_dir}/bert_proteingym_detailed_results.csv"
    
    if [[ ! -f "$json_file" ]]; then
        log_error "Main results file not found: $json_file"
        return 1
    fi
    
    if [[ ! -f "$csv_file" ]]; then
        log_warn "Detailed results file not found: $csv_file"
        log_warn "Some visualizations may be limited"
    fi
    
    return 0
}

install_dependencies() {
    log_info "Checking Python dependencies..."
    
    python -c "
import sys
required_packages = ['matplotlib', 'seaborn', 'pandas', 'numpy', 'scipy']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print('Missing packages:', ', '.join(missing_packages))
    sys.exit(1)
else:
    print('All required packages are available')
    sys.exit(0)
"
    
    if [[ $? -ne 0 ]]; then
        log_info "Installing missing Python packages..."
        pip install matplotlib seaborn pandas numpy scipy
    fi
}

#==============================================================================
# Main Script Logic
#==============================================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Print header
print_header

# Validate required arguments
if [[ -z "$RESULTS_DIR" ]]; then
    log_error "Missing required argument: --results_dir"
    print_usage
    exit 1
fi

# Convert to absolute paths
RESULTS_DIR=$(realpath "$RESULTS_DIR")
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
fi

# Validate inputs
log_info "Validating inputs..."
if ! validate_results_dir "$RESULTS_DIR"; then
    exit 1
fi

# Activate conda environment
log_info "Activating conda environment: $CONDA_ENV_NAME"
eval "$(conda shell.bash hook)"

if check_conda_env "$CONDA_ENV_NAME"; then
    conda activate "$CONDA_ENV_NAME"
    log_info "✅ Activated conda environment: $CONDA_ENV_NAME"
else
    log_warn "Using base conda environment"
fi

# Install dependencies
install_dependencies

# Prepare command
PYTHON_CMD="python ${SCRIPT_DIR}/bert/visualization.py"
PYTHON_CMD+=" --results_dir \"$RESULTS_DIR\""

if [[ -n "$OUTPUT_DIR" ]]; then
    PYTHON_CMD+=" --output_dir \"$OUTPUT_DIR\""
fi

# Display configuration
echo -e "${BLUE}"
echo "📋 Configuration Summary:"
echo "=========================="
echo "Results Directory: $RESULTS_DIR"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "Output Directory:  $OUTPUT_DIR"
else
    echo "Output Directory:  ${RESULTS_DIR}/plots (default)"
fi
echo "Conda Environment: $CONDA_ENV_NAME"
echo "Script Location:   ${SCRIPT_DIR}/bert/visualization.py"
echo -e "${NC}"

# Run visualization
log_info "🚀 Starting BERT results visualization..."
echo -e "${CYAN}Command: $PYTHON_CMD${NC}"
echo ""

if eval "$PYTHON_CMD"; then
    echo ""
    log_info "🎉 Visualization generation completed successfully!"
    
    # Display output location
    if [[ -n "$OUTPUT_DIR" ]]; then
        OUTPUT_LOCATION="$OUTPUT_DIR"
    else
        OUTPUT_LOCATION="${RESULTS_DIR}/plots"
    fi
    
    echo -e "${GREEN}"
    echo "📊 Generated Visualizations:"
    echo "============================"
    if [[ -d "$OUTPUT_LOCATION" ]]; then
        find "$OUTPUT_LOCATION" -name "*.png" -o -name "*.md" | sort | while read -r file; do
            echo "  📈 $(basename "$file")"
        done
    fi
    echo ""
    echo "📁 All files saved to: $OUTPUT_LOCATION"
    echo -e "${NC}"
    
else
    log_error "❌ Visualization generation failed!"
    echo ""
    log_error "Troubleshooting tips:"
    echo "  1. Check that results files exist and are valid"
    echo "  2. Verify Python dependencies are installed"
    echo "  3. Check conda environment activation"
    echo "  4. Ensure sufficient disk space for plots"
    exit 1
fi

echo -e "${PURPLE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    🎉 SUCCESS! 🎉                        ║"
echo "║         BERT Visualization Generation Complete             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"