#!/bin/bash
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
scripts/preparation/download_smolinstruct.sh
nohup python scripts/preparation/preparation_script_molecule_related_nat_lang.py assets/configs/molecules_nl.yaml\
> logs/molecule_related_nat_lang-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &