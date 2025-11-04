#!/bin/bash
source ./src/config/env.sh
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python src/genome_sequence/dataset/prepare_gpt2.py assets/configs/genome_sequence.yaml' > \
    logs/genome_sequence-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
