#!/bin/bash
#
# GuacaMol Dataset Download Script
#
# Usage:
#   LEARNING_SOURCE_DIR="learning_20251104" bash bootstraps/download_guacamol.sh
#

set -e

# Check LEARNING_SOURCE_DIR
if [ -z "$LEARNING_SOURCE_DIR" ]; then
    echo "ERROR: LEARNING_SOURCE_DIR environment variable is not set."
    echo "Please set it before running this script:"
    echo "  export LEARNING_SOURCE_DIR='...'"
    exit 1
fi

echo "Using LEARNING_SOURCE_DIR: $LEARNING_SOURCE_DIR"

# Run the download script
python scripts/preparation/download_guacamol.py

echo ""
echo "GuacaMol download complete!"
echo "You can now run the GPT-2 preparation script:"
echo "  LEARNING_SOURCE_DIR=$LEARNING_SOURCE_DIR python src/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml"
