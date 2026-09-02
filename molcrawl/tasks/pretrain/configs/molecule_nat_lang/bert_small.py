# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import os

from molcrawl.core.paths import get_bert_output_path
from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from molcrawl.data.molecule_nat_lang.utils.vocab_guard import check_vocab_size

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

# 3 epochs of the train split at effective global batch 2560. BERT runs under HF
# Trainer, where the effective batch is batch_size * grad_accum * world_size, so
# 8 * 80 * 4 GPU = 2560 — this REQUIRES the fixed 4-GPU launch (unlike GPT-2,
# whose nanoGPT effective batch is GPU-count-independent).
# 3 * 318,118 train blocks / 2560 = 372.8 -> 373 steps. Verified empirically:
# HF reported epoch=1.61 at 200 steps in the smoke (job 15429).
max_steps = 373
# MLM collapse fix: packing concatenates ~10 documents per 1024 block (measured
# 10.26 EOS per block on the train split); without masking, attention leaks across
# those documents and the run stalls at the unigram level. Confine attention per
# document. Requires the tokenizer to expose EOS as sep_token (see
# data/molecule_nat_lang/utils/tokenizer.py).
document_masking = True

early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "small"  # Choose between small, medium or large
model_path = get_bert_output_path("molecule_nat_lang", model_size)
max_length = 1024
# Shuffled rebuild, the same corpus the GPT-2 ladder switched to in #143. The
# original was written in source order: JS divergence between the head and the
# middle of train measured 0.16449 against a 0.00101 sampling floor, and the
# rebuild brings it to 0.00096. Content is identical (325,752,832 tokens,
# 3,267,172 documents); only the grouping into 1024-token blocks differs.
dataset_dir = MOLECULE_NAT_LANG_DATASET_DIR + "_shuffled"
learning_rate = 0.0001
weight_decay = 0.01
log_interval = 100
save_steps = 1000  # Save checkpoint every 1000 steps instead of 100

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16

# The number max_steps was derived from, stated so the run can check it rather
# than assume it. main.py multiplies per_device x grad_accum x world_size at
# startup and refuses to train if the product differs: under HF the effective
# batch moves with the GPU count, and a 4-GPU request that the scheduler splits
# across 2 nodes would silently train at 1,280.
expected_global_batch = 2560


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



# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 59
