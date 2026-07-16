# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


import os as _os

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 12122
warmup_steps = 242  # ≈ 2 % of max_steps (production spec 2026-07-09、 Phase 1-6 dedup 対応で 249 → 242)
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "large"  # Choose between small, medium or large
model_path = get_bert_output_path("compounds", model_size)
max_length = 128
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# Phase 1-5c (2026-07-16): 5e-5 → 3e-5. The 22913 (5e-5) attempt was
# auto-aborted by the early-plateau detector at eval 6 (val=1.79 > 1.5
# threshold), then 22918 (3e-5) COMPLETED healthy with min val 0.1766
# — matching bert-small 0.176 / bert-medium 0.144. Boss's 2026-07-16
# reply promotes 3e-5 to the unified default across every modality's
# BERT large (compounds / protein / rna / mol_nl) because it's the
# empirically-attested convergent value at 340M scale; 5e-5 would just
# get downgraded again by the same coord ladder, so we skip that hop.
# Env override SUBSET_BERT_LARGE_LR still works for the ladder logic if
# a future attempt needs to try higher or lower.
learning_rate = float(_os.environ.get("SUBSET_BERT_LARGE_LR", "0.00003"))
weight_decay = 0.01
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16
