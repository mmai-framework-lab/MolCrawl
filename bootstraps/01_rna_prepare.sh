#!/bin/bash
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup python scripts/preparation_script_rna.py assets/configs/rna.yaml \
> logs/rna-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &