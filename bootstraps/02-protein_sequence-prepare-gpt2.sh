#!/bin/bash
source ./src/config/env.sh
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python src/protein_sequence/dataset/prepare_gpt2.py assets/configs/protein_sequence.yaml' > \
    logs/protein_sequence-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
