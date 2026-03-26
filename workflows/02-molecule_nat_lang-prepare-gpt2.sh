#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir

# Use local GPT-2 tokenizer (overridable via env var)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export GPT2_TOKENIZER_DIR="${GPT2_TOKENIZER_DIR:-$PROJECT_ROOT/assets/tokenizers/gpt2}"

mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/
nohup $PYTHON molcrawl/molecule_nat_lang/dataset/prepare_gpt2.py assets/configs/molecule_nat_lang_config.yaml \
    > ${LEARNING_SOURCE_DIR}/molecule_nat_lang/logs/molecule_nat_lang-prepare-gpt2-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
