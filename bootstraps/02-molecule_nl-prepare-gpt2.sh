#!/bin/bash
source ./src/config/env.sh
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python src/molecule_nl/dataset/prepare_gpt2.py assets/configs/molecule_nl.yaml' > \
    logs/molecule_nl-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
