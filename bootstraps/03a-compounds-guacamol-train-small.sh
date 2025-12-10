#!/bin/bash
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'python gpt2/train.py gpt2/configs/compounds/train_gpt2_config.py' > \
    logs/compounds-train-small-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
