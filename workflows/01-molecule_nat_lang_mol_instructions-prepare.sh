#!/bin/bash
# Download Mol-Instructions and prepare it for GPT-2 / BERT training.
#
# Steps performed:
#   1. Download zjunlp/Mol-Instructions (Molecule-oriented subset) from HuggingFace
#   2. Run prepare_mol_instructions.py to tokenise, split (80/10/10) and chunk
#      the dataset into training_ready_hf_dataset format.
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>  # must be set beforehand
#   bash workflows/01-molecule_nat_lang_mol_instructions-prepare.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

# Use local GPT-2 tokenizer (overridable via env var)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Offline GPT-2 tokenizer: export only if the directory actually exists.
# Otherwise tokenizer.py falls back to "gpt2" via the HF cache.
if [ -z "${GPT2_TOKENIZER_DIR:-}" ] && [ -d "$PROJECT_ROOT/assets/tokenizers/gpt2" ]; then
    export GPT2_TOKENIZER_DIR="$PROJECT_ROOT/assets/tokenizers/gpt2"
fi


LOG_DIR="${LEARNING_SOURCE_DIR}/molecule_nat_lang/mol_instructions/logs"
mkdir -p "${LOG_DIR}"

# Run via the unified preparation script so --datasets / --download-only flags
# are available.  This also downloads automatically if the source is absent.
echo "[1/1] Preparing Mol-Instructions (download + tokenise if needed)..."
nohup $PYTHON molcrawl/data/molecule_nat_lang/preparation.py \
    assets/configs/molecule_nat_lang_config.yaml \
    --datasets mol_instructions \
    > "${LOG_DIR}/mol_instructions-prepare-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "Preparation running in background. Logs: ${LOG_DIR}/"
