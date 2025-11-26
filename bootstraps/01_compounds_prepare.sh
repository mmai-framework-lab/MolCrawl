#!/bin/bash
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup python scripts/preparation/preparation_script_compounds.py assets/configs/compounds.yaml --force --tokenize-only \
> logs/compounds-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &