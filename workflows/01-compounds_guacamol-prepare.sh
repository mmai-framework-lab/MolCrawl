#!/bin/bash
# Download GuacaMol benchmark SMILES files for GPT-2 pre-training on compounds.
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/01-compounds_guacamol-prepare.sh

set -e

# Load common functions (sets $PYTHON)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

echo "Using LEARNING_SOURCE_DIR: $LEARNING_SOURCE_DIR"

# Run the download script
$PYTHON molcrawl/data/compounds/download_guacamol.py

echo ""
echo "GuacaMol download complete!"
echo "You can now run the GPT-2 preparation script:"
echo "  bash workflows/02-compounds-prepare-gpt2.sh"
