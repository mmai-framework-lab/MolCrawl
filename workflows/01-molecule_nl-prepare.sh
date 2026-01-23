#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nl/logs/
scripts/preparation/download_smolinstruct.sh
nohup python scripts/preparation/preparation_script_molecule_related_nat_lang.py assets/configs/molecules_nl.yaml\
> ${LEARNING_SOURCE_DIR}/molecule_nl/logs/molecule_related_nat_lang-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &