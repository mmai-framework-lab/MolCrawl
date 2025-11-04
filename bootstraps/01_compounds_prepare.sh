#!/bin/bash
source ./src/config/env.sh
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup python scripts/preparation_script_compounds.py assets/configs/compounds.yaml \
> logs/compounds-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &