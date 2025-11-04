#!/bin/bash
LEARNING_SOURCE_DIR="learning_source_20251020-molecule-nl"
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs $LEARNING_SOURCE_DIR
LEARNING_SOURCE_DIR=$LEARNING_SOURCE_DIR nohup python scripts/preparation_script_molecule_related_nat_lang.py assets/configs/molecules_nl.yaml \
> logs/molecule_related_nat_lang-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &