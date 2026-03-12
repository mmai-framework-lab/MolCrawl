# BERT (small) fine-tuning config for Mol-Instructions
#
# Continues from the molecule_nat_lang BERT pretraining checkpoint using the
# Mol-Instructions fine-tuning dataset.
#
# Based on molcrawl/bert/configs/molecule_nat_lang.py — only the dataset
# path and model output directory differ.


from molcrawl.config.paths import MOL_INSTRUCTIONS_DATASET_DIR, get_bert_output_path
from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

# bert/main.py looks for a `tokenizer` variable in globals() to build the
# DataCollatorForLanguageModeling.  MoleculeNatLangTokenizer wraps the actual
# HuggingFace tokenizer in .tokenizer; main.py handles that unwrapping.
tokenizer = MoleculeNatLangTokenizer()

# Use the vocab size reported by MoleculeNatLangTokenizer.
# - CodeLlama-7b-hf: 32016
# - Offline fallback (MinimalTokenizer): 50002  ← used for data preparation in
#   this environment because the network is unavailable.
# Set to 50002 so the embedding table covers all token IDs in the dataset.
meta_vocab_size = 50002

model_size = "small"
# Fine-tuning checkpoint output — separate from pretraining output
model_path = get_bert_output_path("molecule_nat_lang_mol_instructions", model_size)

max_length = 1024
dataset_dir = MOL_INSTRUCTIONS_DATASET_DIR

# Fine-tuning hyper-parameters (lower LR and fewer steps than pretraining)
learning_rate = 1e-5
max_steps = 60000  # ~10 % of the 600k pretraining steps
weight_decay = 1e-1

log_interval = 100
save_steps = 100

batch_size = 8
per_device_eval_batch_size = 1
gradient_accumulation_steps = 5 * 16


def preprocess_function(examples):
    """Add attention_mask to the dataset"""
    if "input_ids" in examples:
        attention_masks = []
        for input_ids in examples["input_ids"]:
            attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
            attention_masks.append(attention_mask)
        examples["attention_mask"] = attention_masks
    return examples


# Special Tokens (CodeLlama tokenizer)
start_instruction = 1
end_instruction = [518, 29914, 25580, 29962]
eos_token = 2  # eos
