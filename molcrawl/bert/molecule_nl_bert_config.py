"""
Configuration for BERT training on Molecule NL dataset

Usage:
    python bert/main.py bert/molecule_nl_bert_config.py
"""

from transformers import AutoTokenizer
from molcrawl.utils.environment_check import check_learning_source_dir

# Add parent directory to path

# 共通環境チェックモジュールを追加

# Model configuration
model_size = "small"  # Options: "small", "medium", "large"
max_length = 512  # Maximum sequence length for BERT

# Dataset configuration
learning_source_dir = check_learning_source_dir()
dataset_dir = f"{learning_source_dir}/molecule_nl/arrow_splits"

# Training configuration
model_path = f"runs_train_bert_molecule_nl_{model_size}"
learning_rate = 6e-5
weight_decay = 1e-1
warmup_steps = 1000
max_steps = 100000
batch_size = 16
gradient_accumulation_steps = 4
per_device_eval_batch_size = 16
log_interval = 100

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
meta_vocab_size = (len(tokenizer) // 8 + 1) * 8

# BERT doesn't need custom preprocessing since data already has attention_mask
# Data is already in the correct format with input_ids and attention_mask

print("📊 BERT Molecule NL Configuration")
print(f"   Dataset: {dataset_dir}")
print(f"   Model: {model_size}")
print(f"   Vocab size: {meta_vocab_size}")
print(f"   Max length: {max_length}")
print(f"   Batch size: {batch_size}")
print(f"   Output: {model_path}")
