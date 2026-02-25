#!/bin/bash
# 化合物データセット準備ワークフロー

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Create log directory
mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs/

# Run preparation script
echo "🚀 Starting compounds dataset preparation..."
echo "📁 Learning source: ${LEARNING_SOURCE_DIR}"

nohup python src/preparation/preparation_script_compounds.py \
    assets/configs/compounds.yaml \
    --force \
    > ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &

echo "✅ Preparation started in background (check logs for progress)"
