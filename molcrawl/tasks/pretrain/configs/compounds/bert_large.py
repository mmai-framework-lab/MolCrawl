# compounds BERT large — packed 1024 ladder (v4 data, 2026-08-05)
# launch: torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py <this config>


import os as _os

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

# v4 packed data (2026-08-05): train = 398,917 blocks x 1024, no padding.
# HF Trainer is per-device, so global batch = batch_size * grad_accum * n_GPUs
# = 8 * 80 * 4 = 2,560 seq (assumes the 4-GPU launch used by the whole ladder).
# 10 epochs at 2,560 = floor(10 * 398,917 / 2560) = 1,558 steps.
max_steps = 1558
warmup_steps = 31  # ~2 % of max_steps; < max_steps so LR reaches peak
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "large"  # Choose between small, medium or large
model_path = get_bert_output_path("compounds", model_size)
max_length = 1024  # packed blocks; sets BertConfig.max_position_embeddings
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# The compounds sets are packed in source-parquet order, so the split's leading rows
# are shorter and easier than the split as a whole. Draw the eval subset at random
# instead. Off by default in main.py because protein / RNA / genome shuffle in prep and
# gain nothing from it.
eval_subset_random = True
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
log_interval = 50  # = eval_steps -> ~31 eval points over the run
save_steps = 100  # must be a multiple of eval_steps for load_best_model_at_end

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 7
