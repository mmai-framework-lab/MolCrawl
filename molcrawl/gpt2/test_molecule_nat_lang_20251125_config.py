"""
Quick test configuration for GPT-2 training on Molecule NL dataset (learning_20251125)
This is a minimal test to verify the training pipeline works correctly.
"""

import os
import sys

# Add parent directory to path

# I/O
out_dir = "test_gpt2_molecule_nat_lang_20251125"
eval_interval = 10  # Evaluate very frequently for testing
log_interval = 1
eval_iters = 5  # Just a few iterations for quick eval
eval_only = False
always_save_checkpoint = False
init_from = "scratch"

# Data
learning_source_dir = "learning_20251125"
dataset = "molecule_nat_lang"

# Configure dataset parameters for PreparedDataset
dataset_params = {"dataset_dir": f"{learning_source_dir}/molecule_nat_lang/arrow_splits"}

# Model - Very small for quick testing
block_size = 128  # Small context for fast training
n_layer = 4  # Minimal layers
n_head = 4  # Minimal heads
n_embd = 256  # Small embedding
dropout = 0.1
bias = False

# Training - Minimal iterations for testing
gradient_accumulation_steps = 1
batch_size = 4  # Very small batch
learning_rate = 3e-4
max_iters = 50  # Just 50 iterations to verify it works
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 5
lr_decay_iters = 50
min_lr = 3e-5

# System
device = "cuda"
dtype = "bfloat16"
compile = False

# Vocab size (Llama-2-7b-hf has 32000 tokens, rounded to 32008)
meta_vocab_size = 32008

# Tensorboard
tensorboard = False

print("=" * 70)
print("GPT-2 Molecule NL Test Configuration (learning_20251125)")
print("=" * 70)
print(f"Dataset: {dataset_params['dataset_dir']}")
print(f"Vocab size: {meta_vocab_size}")
print(f"Block size: {block_size}")
print(f"Model: {n_layer} layers, {n_head} heads, {n_embd} embd")
print(f"Batch size: {batch_size}")
print(f"Max iterations: {max_iters}")
print(f"Output: {out_dir}")
print("=" * 70)
print("This is a quick test run to verify the training pipeline.")
print("=" * 70)
