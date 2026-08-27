# config for training BERT on protein sequences using ESM tokenizer
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ python bert/main.py bert/configs/protein_sequence_large.py


import os as _os
from typing import Any, Dict, List

import torch
from transformers import DataCollatorForLanguageModeling

from molcrawl.core.paths import UNIPROT_DATASET_DIR, get_bert_output_path
from molcrawl.data.protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer

# Tokenizer instantiation - BERT compatible ESM tokenizer
tokenizer = create_bert_protein_tokenizer()


# Dataset preprocessing function to add attention_mask
def preprocess_function(examples):
    """
    Add attention_mask to dataset for BERT compatibility
    """
    # Handle batch processing
    if "input_ids" in examples:
        input_ids = examples["input_ids"]

        # Create attention_mask (1 for real tokens, 0 for padding)
        if isinstance(input_ids[0], list):  # Batch of sequences
            attention_masks = []
            for seq in input_ids:
                # Assume padding token is 0 or tokenizer.pad_token_id
                pad_token_id = (
                    tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
                )
                attention_mask = [1 if token != pad_token_id else 0 for token in seq]
                attention_masks.append(attention_mask)
            examples["attention_mask"] = attention_masks
        else:  # Single sequence
            pad_token_id = (
                tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
            )
            examples["attention_mask"] = [1 if token != pad_token_id else 0 for token in input_ids]

    return examples


# Custom data collator that handles the tokenizer compatibility
class ProteinSequenceDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for protein sequences that handles field name conversion
    """

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Override to handle any remaining field name issues
        """
        # Convert any sequence_tokens to input_ids before processing
        for example in examples:
            if "sequence_tokens" in example and "input_ids" not in example:
                example["input_ids"] = example.pop("sequence_tokens")

        # Call parent method
        return super().torch_call(examples)


# Use custom data collator
data_collator = ProteinSequenceDataCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

# Training configuration
max_steps = 11177  # 3 epochs of train at global batch 2560 = floor(3*9,538,464/2560)
early_stopping = False  # Pretraining: run the full schedule, no early stopping
# MLM collapse fix: packing concatenates ~3-5 proteins per 1024 block; without
# masking, attention leaks across those documents. Confine attention per document.
document_masking = True
# Collapse detector OFF for protein (boss decision 2026-08-26). The threshold was the
# analytical [MASK] unigram baseline (H=2.8947 over 24 amino acids * 0.9089 non-copy
# ratio = 2.631) and is itself clean, but a fixed level-line stop cannot tell a slowly
# descending run from a stalled one, so it would kill a healthy slow-descent run before
# it crosses. Same conclusion reached for genome, where it was turned off. Hold large
# until the small recipe (fixed optimizer + re-measured LR) is settled.
degenerate_loss_threshold = None
model_size = "large"  # Choose between small, medium or large
model_path = get_bert_output_path("protein_sequence", model_size)
max_length = 1024
dataset_dir = UNIPROT_DATASET_DIR
# Phase 1-5c (2026-07-16): 5e-5 → 3e-5. compounds bert-large retrain at
# 3e-5 (jobid 22918) completed healthy at min val 0.1766. Boss aligns
# every modality's BERT large to 3e-5 to skip the coord ladder's
# 5e-5 → 3e-5 auto-downgrade hop.
learning_rate = float(_os.environ.get("SUBSET_BERT_LARGE_LR", "0.00003"))
weight_decay = 0.01
log_interval = 100
save_steps = 1000  # Save checkpoint every 1000 steps instead of 100

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16

# Protein sequence specific vocabulary size
# ESM tokenizer uses character-level tokenization for protein sequences
meta_vocab_size = len(tokenizer.get_vocab())

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 77
