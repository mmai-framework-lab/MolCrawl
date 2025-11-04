#!/bin/bash
source ./src/config/env.sh
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python gpt2/train.py ./gpt2/data/rna/train_gpt2_config.py' > \
    logs/rna-train-small-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
