# BERT (small) fine-tuning config for ChEMBL
#
# Continues from the compounds BERT pretraining checkpoint using the ChEMBL
# fine-tuning dataset (canonical SMILES from ChEMBL 36).
#
# Based on molcrawl/tasks/pretrain/configs/compounds/bert_small.py — only the dataset path,
# model output directory, learning rate and max_steps differ.

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import CHEMBL_DATASET_DIR, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

model_size = "small"
# Fine-tuning checkpoint output — separate from pretraining output
model_path = get_bert_output_path("compounds_chembl", model_size)
# Pretraining checkpoint to initialise weights from when no fine-tune checkpoint exists.
pretrain_model_path = get_bert_output_path("compounds", model_size)

max_length = 256
dataset_dir = CHEMBL_DATASET_DIR

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


def preprocess_function(examples):
    """Map compounds token columns to BERT inputs."""
    if "input_ids" not in examples and "tokens" in examples:
        examples["input_ids"] = examples["tokens"]

    if "input_ids" in examples and "attention_mask" not in examples:
        pad_token_id = 0
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        examples["attention_mask"] = [
            [1 if token_id != pad_token_id else 0 for token_id in input_ids] for input_ids in examples["input_ids"]
        ]

    return examples
