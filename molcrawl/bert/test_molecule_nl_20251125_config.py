"""
Test Configuration for BERT training on Molecule NL dataset (learning_20251125)

Usage:
    python bert/main.py bert/test_molecule_nl_20251125_config.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

# Add parent directory to path

# Model configuration
model_size = "small"  # Small model for quick testing
max_length = 128  # Shorter context for faster testing

# Dataset configuration
learning_source_dir = "learning_20251125"
dataset_dir = f"{learning_source_dir}/molecule_nl/arrow_splits"

# Training configuration (minimal for testing)
model_path = "test_bert_molecule_nl_20251125"
learning_rate = 6e-5
weight_decay = 1e-1
warmup_steps = 10  # Reduced for testing
max_steps = 50  # Just 50 steps for validation
batch_size = 4  # Small batch for quick testing
gradient_accumulation_steps = 1
per_device_eval_batch_size = 4
log_interval = 10  # Log more frequently for testing

# Tokenizer setup - hardcode vocab size to avoid auth issues
meta_vocab_size = 32008  # Llama-2-7b-hf vocab size

# BERT-specific settings
mlm_probability = 0.15  # Standard MLM masking probability


# Custom data collator for MLM without tokenizer
@dataclass
class CustomDataCollatorForMLM:
    """Custom data collator for Masked Language Modeling without tokenizer"""

    mlm_probability: float = 0.15
    max_length: int = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find max length in this batch
        batch_max_length = min(max(len(f["input_ids"]) for f in features), self.max_length)

        # Pad/truncate all sequences to batch_max_length
        input_ids_list = []
        attention_mask_list = []

        for f in features:
            ids = f["input_ids"][:batch_max_length]  # truncate if needed
            mask = f["attention_mask"][:batch_max_length]

            # Pad if needed
            padding_length = batch_max_length - len(ids)
            if padding_length > 0:
                ids = ids + [0] * padding_length
                mask = mask + [0] * padding_length

            input_ids_list.append(ids)
            attention_mask_list.append(mask)

        # Convert to tensors
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        # Create labels (clone input_ids before masking)
        labels = input_ids.clone()

        # Create random mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Don't mask padding tokens
        masked_indices = masked_indices & (attention_mask.bool())

        # Set labels to -100 for non-masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with [MASK] token (id=4)
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = 4  # [MASK] token id

        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(meta_vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time, keep original token

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


data_collator = CustomDataCollatorForMLM(mlm_probability=mlm_probability, max_length=max_length)

print("📊 BERT Molecule NL Test Configuration")
print(f"   Dataset: {dataset_dir}")
print(f"   Model: {model_size}")
print(f"   Vocab size: {meta_vocab_size}")
print(f"   Max length: {max_length}")
print(f"   Batch size: {batch_size}")
print(f"   Max steps: {max_steps}")
print(f"   MLM probability: {mlm_probability}")
print(f"   Output: {model_path}")
