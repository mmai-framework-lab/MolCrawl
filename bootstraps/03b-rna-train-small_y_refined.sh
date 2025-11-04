#!/bin/bash
export LEARNING_SOURCE_DIR="learning_source_20250818"
cd /wren/matsubara/riken-dataset-fundational-model
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 python gpt2/train.py gpt2/configs/rna/train_gpt2_config_yigarashi.py' > \
    logs/rna-yigarashi-train-small-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &