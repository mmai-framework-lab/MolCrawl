"""
Configuration for GPT-2 training on Molecule NL dataset

Usage:
    python gpt2/train.py --config=gpt2/molecule_nl_gpt2_config.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer

# I/O
out_dir = "runs_train_gpt2_molecule_nl"
eval_interval = 2000
log_interval = 100
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# Data
learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR", "learning_20251121")
dataset = "molecule_nl"

# Configure dataset parameters for PreparedDataset
dataset_params = {"dataset_dir": f"{learning_source_dir}/molecule_nl/arrow_splits"}

# Model
block_size = 512  # Context length
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = False

# Training
gradient_accumulation_steps = 8
batch_size = 16
learning_rate = 3e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 100000
min_lr = 3e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = False

# Tokenizer (for vocab size)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
meta_vocab_size = (len(tokenizer) // 8 + 1) * 8

# Tensorboard
tensorboard = True
tensorboard_dir = f"{out_dir}/tensorboard"

print("📊 GPT-2 Molecule NL Configuration")
print(f"   Dataset: {dataset_params['dataset_dir']}")
print(f"   Vocab size: {meta_vocab_size}")
print(f"   Block size: {block_size}")
print(f"   Batch size: {batch_size}")
print(f"   Gradient accumulation: {gradient_accumulation_steps}")
print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
print(f"   Output: {out_dir}")
