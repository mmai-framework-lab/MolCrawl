#!/bin/bash

# RNA GPT-2 Training Script for Yigarashi Data
# Test run for the modified train.py

echo "🧬 Starting RNA GPT-2 Training with Yigarashi Data..."
echo "📁 Data Directory: /wren/matsubara/riken-dataset-fundational-model/yigarashi-rna-2025-10-07/rna/"
echo "🎯 Configuration: gpt2/configs/rna/train_gpt2_config_yigarashi.py"

# Set the config file
CONFIG_FILE="gpt2/configs/rna/train_gpt2_config_yigarashi.py"

# Test run with small iterations first
python src/gpt2/train.py \
    --config=$CONFIG_FILE \
    --max_iters=100 \
    --eval_interval=50 \
    --log_interval=10 \
    --batch_size=2 \
    --eval_only=False

echo "✅ Training completed or stopped."
