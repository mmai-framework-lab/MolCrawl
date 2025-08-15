# config for training BERT on protein sequences using ESM tokenizer
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ python bert/main.py bert/configs/protein_sequence.py

from protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer
from transformers import DataCollatorForLanguageModeling
from typing import Dict, List, Any
import torch


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
                pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0
                attention_mask = [1 if token != pad_token_id else 0 for token in seq]
                attention_masks.append(attention_mask)
            examples["attention_mask"] = attention_masks
        else:  # Single sequence
            pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0
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
max_steps = 600000
model_path = "runs_train_bert_protein_sequence"
max_length = 1024
dataset_dir = "learning_source_202508/uniprot/training_ready_hf_dataset"
learning_rate = 6e-6
weight_decay = 1e-1
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 1

gradient_accumulation_steps = 5 * 16
output_dir = "out-bert-protein-sequence"

# Model size configuration
# Choose between small, medium or large
model_size = "small"

# Protein sequence specific vocabulary size
# ESM tokenizer uses character-level tokenization for protein sequences
meta_vocab_size = len(tokenizer.get_vocab())
