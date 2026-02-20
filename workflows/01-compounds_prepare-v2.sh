#!/bin/bash
# 化合物データセット準備ワークフロー（改訂版）

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Create log directory
mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs/

# Run new preparation script
echo "🚀 Starting compounds dataset preparation (v2)..."
echo "📁 Learning source: ${LEARNING_SOURCE_DIR}"

python scripts/preparation/preparation_script_compounds_v2.py \
    assets/configs/compounds.yaml \
    > ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-preparation-v2-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1

echo "✅ Preparation completed!"
