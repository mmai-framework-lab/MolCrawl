#!/bin/bash
source ./src/config/env.sh
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup python scripts/preparation_script_molecule_related_nat_lang.py assets/configs/molecules_nl.yaml \
> logs/molecule_related_nat_lang-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &