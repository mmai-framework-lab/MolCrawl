# BERT (large) fine-tuning config for ProteinGym DMS sequences
#
# Continues from the protein_sequence BERT pretraining checkpoint using the
# ProteinGym fine-tuning dataset (mutated + wild-type sequences from DMS assays).
#
# Based on bert/configs/protein_sequence.py — key differences:
#   - dataset_dir / model_path point to the ProteinGym dataset and output
#   - max_steps reduced to 60000 (fine-tuning, not pretraining from scratch)
#   - learning_rate reduced to 1e-5


from typing import Any, Dict, List

import torch
from transformers import DataCollatorForLanguageModeling

from molcrawl.config.paths import PROTEINGYM_DATASET_DIR, get_bert_output_path
from molcrawl.protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer

# Tokenizer instantiation - BERT compatible ESM tokenizer
tokenizer = create_bert_protein_tokenizer()


def preprocess_function(examples):
    """Add attention_mask to dataset for BERT compatibility."""
    if "input_ids" in examples:
        pad_token_id = (
            tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
        )
        if isinstance(examples["input_ids"][0], list):
            examples["attention_mask"] = [[1 if t != pad_token_id else 0 for t in seq] for seq in examples["input_ids"]]
        else:
            examples["attention_mask"] = [1 if t != pad_token_id else 0 for t in examples["input_ids"]]
    return examples


class ProteinSequenceDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that converts sequence_tokens → input_ids if needed."""

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        for example in examples:
            if "sequence_tokens" in example and "input_ids" not in example:
                example["input_ids"] = example.pop("sequence_tokens")
        return super().torch_call(examples)


data_collator = ProteinSequenceDataCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

# Training configuration
model_size = "large"
model_path = get_bert_output_path("protein_sequence_proteingym", model_size)
# Pretraining checkpoint to initialise weights from when no fine-tune checkpoint exists.
pretrain_model_path = get_bert_output_path("protein_sequence", model_size)

max_length = 1024
dataset_dir = PROTEINGYM_DATASET_DIR

# Fine-tuning hyper-parameters (lower LR and fewer steps than pretraining)
learning_rate = 1e-5
max_steps = 60000  # ~10 % of the 600k pretraining steps
weight_decay = 1e-1

log_interval = 100
save_steps = 1000
early_stopping_patience = 3  # Stop after 3 evals (300 steps) with no improvement

batch_size = 8
per_device_eval_batch_size = 8
gradient_accumulation_steps = 5 * 16

# Protein sequence vocabulary size (EsmSequenceTokenizer: 33 character-level tokens)
meta_vocab_size = len(tokenizer.get_vocab())
