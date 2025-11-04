#!/bin/bash
export LEARNING_SOURCE_DIR="learning_source_20251020-molecule-nl"
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python gpt2/train.py ./gpt2/configs/molecule_nl/train_gpt2_large_config.py' > \
    logs/molecule_nl-train-large-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
