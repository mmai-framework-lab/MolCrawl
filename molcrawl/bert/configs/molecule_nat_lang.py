# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import os

from molcrawl.core.paths import get_bert_output_path
from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from molcrawl.molecule_nat_lang.utils.vocab_guard import check_vocab_size

# Get LEARNING_SOURCE_DIR from environment variable directly
LEARNING_SOURCE_DIR = os.environ.get("LEARNING_SOURCE_DIR", "./learning_source_20260105-molecule-nl")
MOLECULE_NAT_LANG_DIR = LEARNING_SOURCE_DIR + "/molecule_nat_lang"
MOLECULE_NAT_LANG_DATASET_DIR = MOLECULE_NAT_LANG_DIR + "/training_ready_hf_dataset"

tokenizer = Tokenizer()

# molecule_nat_lang uses the GPT-2 tokenizer (vocab_size=50257). Pad up to
# the next multiple of 8 for efficient embedding lookups. check_vocab_size()
# verifies the result matches the value baked into existing checkpoints so
# a tokenizer swap is caught at startup rather than silently trashing weights.
meta_vocab_size = (tokenizer.vocab_size // 8 + 1) * 8
check_vocab_size(meta_vocab_size)

max_steps = 60000
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "small"  # Choose between small, medium or large
model_path = get_bert_output_path("molecule_nat_lang", model_size)
max_length = 1024
dataset_dir = MOLECULE_NAT_LANG_DATASET_DIR
learning_rate = 6e-6
weight_decay = 1e-1
log_interval = 100
save_steps = 1000  # Save checkpoint every 1000 steps instead of 100

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16


# Add preprocessing function to create attention_mask
def preprocess_function(examples):
    """Add attention_mask to the dataset"""
    if "input_ids" in examples:
        # Create attention_mask: 1 for real tokens, 0 for padding
        attention_masks = []
        for input_ids in examples["input_ids"]:
            # Assuming pad_token_id is 0
            attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
            attention_masks.append(attention_mask)

        examples["attention_mask"] = attention_masks

    return examples


