#!/bin/bash
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'python gpt2/train.py gpt2/configs/compounds/train_gpt2_xl_config.py' > \
    logs/compounds-train-xl-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &