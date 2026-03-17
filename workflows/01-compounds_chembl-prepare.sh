#!/bin/bash
# Download ChEMBL 36 from EBI FTP and prepare it for GPT-2 / BERT fine-tuning.
#
# Steps performed:
#   1. Download ChEMBL 36 SQLite archive from ftp.ebi.ac.uk (~4.4 GB)
#   2. Extract canonical SMILES from the compound_structures table
#   3. Tokenise with CompoundsTokenizer (SMILES regex + WordPiece, vocab_size=256)
#   4. Split 80/10/10 (train/valid/test) and save as HuggingFace Dataset
#
# Prerequisites:
#   - ~10 GB free disk space (archive + SQLite + dataset)
#   - Internet access to ftp.ebi.ac.uk
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/01-compounds_chembl-prepare.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/chembl/logs"
mkdir -p "${LOG_DIR}"

echo "[1/1] Preparing ChEMBL fine-tuning dataset (download + tokenise if needed)..."
nohup $PYTHON molcrawl/preparation/preparation_script_compounds.py \
    assets/configs/compounds.yaml \
    --datasets chembl_finetune \
    > "${LOG_DIR}/chembl-prepare-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "Preparation running in background. Logs: ${LOG_DIR}/"
