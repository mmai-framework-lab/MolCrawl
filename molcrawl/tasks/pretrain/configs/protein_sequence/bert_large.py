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
# 9 epochs at global batch 2,560. At 3 epochs (11,177 steps) the loss is still falling
# and sizes cannot be compared (protein-bert-grid-verdict-2026-09-16 §2; all-bert-order
# 2026-09-17 §6.1). 33,531 is 3 x 11,177, the length the protein GPT-2 9-epoch runs used;
# an exact floor(9 x 9,538,464 / 2,560) would be 33,533.
max_steps = 33531
# 10 % of max_steps. Written here rather than in the grid: without this line the base
# fell to main.py's default of 200, and a grid that set only its own warmup would have
# gone to 3.3 % the moment max_steps changed here.
warmup_steps = 3353
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
# 0.999, not main.py's default of 0.95. The note above names adam_beta2=0.95 with a
# 200-step warmup as the cause of the collapse every learning rate fell into, and this
# base declared neither, so running it directly reproduced exactly that setting
# (all-bert-order-2026-09-17 §3.3). warmup_steps is already set above. main.py's
# default stays 0.95: the other modalities run on it and have not collapsed.
adam_beta2 = 0.999
log_interval = 100
save_steps = 1000  # Save checkpoint every 1000 steps instead of 100

# Keep the checkpoint the reported number came from. Evaluation is 10x finer than
# saving here (100 against 1,000), so the minimum lands off the save grid nine
# times in ten and best_model_checkpoint points at a neighbour instead.
save_on_improve = True

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16
# Where the input is fetched, not what is computed: the same rows in the same
# order, pulled by four worker processes into pinned buffers instead of by the
# training process itself. main.py defaults both off (main.py:655-656), and with
# them off the Arrow read, the MLM draw and the document masking all sit on the
# critical path of every step. Measured on rna small at 8 x 80, 4 GPUs: in fp32
# these two settings alone take a step from 12.476 to 11.044 s (1.13x); with bf16
# as well it falls to 3.812 s (3.27x from where it started). bf16 without them is
# 0.94x. Neither half does much on its own.
#
# The masked positions are not the same as a 0-worker run: the collate runs in
# the worker, whose RNG PyTorch seeds per worker. The rate and the objective are
# unchanged and each setting reproduces itself, but the draw differs, so a run
# started with workers is not a continuation of one started without.
#
# Workers and pinning approved 2026-09-16 (all-bert-throughput-verdict-2026-09-16b);
# bf16 approved 2026-09-17 (all-bert-order-2026-09-17 §1). bf16 is the one of the three
# that changes numerics: every GPT-2 result and genome's BERT (bert_small_subset.py)
# already ran in bf16, so this aligns the remaining BERT modalities with them rather
# than introducing a new precision. Results from before this line ran in fp32 and are
# tabulated with a precision column (§1.4).
dataloader_num_workers = 4
dataloader_pin_memory = True
bf16 = True

# 8 x 80 x 4 GPUs = 2,560, the batch max_steps was derived from. Without this line
# main.py's global-batch check does not run, and a launch on a different GPU count
# trains at a different batch without stopping (all-bert-order-2026-09-17 §5.3).
expected_global_batch = 2560

# Protein sequence specific vocabulary size
# ESM tokenizer uses character-level tokenization for protein sequences
meta_vocab_size = len(tokenizer.get_vocab())

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 77
