#!/bin/bash
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup python scripts/preparation/preparation_script_protein_sequence.py assets/configs/protein_sequence.yaml \
> logs/protein-sequence-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &