# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 12412
warmup_steps = 249  # ≈ 2 % of max_steps (production spec 2026-07-09)
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "large"  # Choose between small, medium or large
model_path = get_bert_output_path("compounds", model_size)
max_length = 128
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# Phase 1-5b (2026-07-15): 1e-4 → 5e-5. The first retrain at 1e-4
# (jobid 22889) reproduced the 07-13 divergence: val_loss stuck at
# 2.5x for 8 evals (identical to 07-13 pre-divergence pattern) while
# small/medium at the SAME LR descended to 0.8x by eval 4. Model-size
# LR sensitivity for BERT-large (340M) — ALBERT/RoBERTa land in
# 3e-5..5e-5 for large, 1e-4 is a base-size value. 5e-5 unified across
# every modality's BERT large (compounds / protein / rna / mol_nl).
# Env override SUBSET_BERT_LARGE_LR keeps the LR-ladder auto-downgrade
# (5e-5 → 3e-5 → 1e-5) machinery working without editing this file
# between attempts.
import os as _os
learning_rate = float(_os.environ.get("SUBSET_BERT_LARGE_LR", "0.00005"))
weight_decay = 0.01
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16
