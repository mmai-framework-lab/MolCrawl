"""Genome GPT-2 (small) — single-nucleotide / subset (Evo2 species) flow.

Reference config for training GPT-2 on the Phase 3 parquet produced by the
subset pipeline (``parquet_gpt2/<accession>.parquet``, 1024-token causal-LM
rows of nucleotide ids only).

Differences from :mod:`gpt2_small`:
- ``meta_vocab_size = 10`` (A/T/G/C/N + PAD/UNK/CLS/SEP/MASK).
- Tokenizer is the character-level HF tokenizer; the GPT-2 trainer only
  consults it for ambiguous-token resolution (off by default), so it is here
  mainly for parity and decoding helpers.
- ``dataset_dir`` points at the subset's ``training_ready_hf_dataset_gpt2/``
  Arrow DatasetDict (Phase 6 output); subset comes from the ``GENOME_SUBSET``
  env var.
- ``out_dir`` / ``tensorboard_dir`` are suffixed with the subset name.
"""

import os

from transformers import AutoTokenizer

from molcrawl.core.paths import (
    GENOME_SEQUENCE_DIR,
    get_custom_tokenizer_path,
    get_gpt2_output_path,
)
from molcrawl.data.genome_sequence.utils.single_nuc_tokenizer import (
    build_single_nuc_tokenizer,
)

# ---- subset selection ----------------------------------------------------- #
GENOME_SUBSET = os.environ.get("GENOME_SUBSET")
if not GENOME_SUBSET:
    raise RuntimeError(
        "GENOME_SUBSET env var is required for the subset GPT-2 config. "
        "Example: GENOME_SUBSET=mammal_centered python -m molcrawl.models.gpt2.train ..."
    )

# ---- paths ---------------------------------------------------------------- #
# GPT2_LR_TAG (optional, set during LR sweeps) makes sweep checkpoint dirs
# distinct so concurrent runs do not overwrite each other. Empty for production.
_lr_tag = os.environ.get("GPT2_LR_TAG", "")
_subset_suffix = f"-{GENOME_SUBSET}" + (f"-{_lr_tag}" if _lr_tag else "")
out_dir = get_gpt2_output_path("genome_sequence", "small") + _subset_suffix
tensorboard_dir = out_dir
dataset_dir = f"{GENOME_SEQUENCE_DIR}/{GENOME_SUBSET}/training_ready_hf_dataset_gpt2"

# ---- tokenizer (10-symbol single-nucleotide; trainer barely uses it) ----- #
_custom_tokenizer_path = get_custom_tokenizer_path("genome_sequence", "bert_single_nuc")
build_single_nuc_tokenizer(_custom_tokenizer_path)  # idempotent build/cache
tokenizer = AutoTokenizer.from_pretrained(_custom_tokenizer_path)
meta_vocab_size = len(tokenizer)  # = 10

tensorboard = True

# ---- training hyperparameters (mirror gpt2_small) ------------------------ #
# Phase 1-5 (2026-07-14): batch/grad_accum aligned to global batch 2,560
# per charter (per_device 8 × grad_accum 80 × n_GPUs 4 = 2,560). Previous
# 12 × 40 × 4 = 1,920 was Phase 0 vintage. block_size stays 1024 (matches
# Phase 3 gpt2_chunk_size, unchanged).
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 5 * 16

# LR pre-check on the pre-G2 (2026-05) mammal_centered dataset (jobs
# 19068-19070) showed 6e-4 spiked back up around step 6k and 6e-5 was
# monotone. That result is documented here for reference; the shipped
# default follows the 2026-07-09 production spec (GPT-3 ladder: 6e-4 for
# small). The G2 dataset (contig-split, 83M-window budget) is not identical
# to the flow that spiked, so revisiting the LR should be a smoke/short-run
# call rather than a blind rollback. The readiness report flags this so the
# call is made explicitly before the 42-config run launches.
# GPT2_LR_TAG env var stays as the escape hatch for sweeps.
learning_rate = float(os.environ.get("SUBSET_GPT2_LR", "0.0006"))
min_lr = learning_rate / 10  # → 10% of peak per Chinchilla / GPT-2 convention

# Compute-matched schedule from the realized dataset row count (charter
# Condition 2 output): global batch 2,560 × 3 epochs per subset. Reading
# via `load_from_disk` is memory-mapped, so this only touches metadata.
_GLOBAL_BATCH = 2560
_N_EPOCH = 3
from datasets import load_from_disk as _load
_ds_for_len = _load(dataset_dir)
_train_n = len(_ds_for_len["train"])
max_iters = (_N_EPOCH * _train_n + _GLOBAL_BATCH - 1) // _GLOBAL_BATCH
lr_decay_iters = max_iters
warmup_iters = max(int(0.02 * max_iters), 100)  # ≈ 2 % of max_iters
del _ds_for_len

# Fixed-schedule comparison run (charter §「比較系は early_stopping OFF、
# compute-matched」).  gpt2/train.py reads this via globals().get.
early_stopping = False

eval_interval = 1000
eval_iters = 200
log_interval = 10

# "scratch" for first pretraining run; flip to "resume" on subsequent
# resumes from out_dir/<checkpoint>/. Previously this was hard-coded to
# "resume" (mirroring the legacy gpt2_small.py), which crashes the trainer
# on a fresh subset run because no checkpoint exists yet.
init_from = "scratch"
always_save_checkpoint = True
save_checkpoint_steps = None
max_checkpoints = 5

weight_decay = 0.1

dataset = "genome_sequence"
dataset_params = {"dataset_dir": dataset_dir}

# Special token ids (single-nucleotide vocab from single_nuc_tokenizer):
#   PAD=5, UNK=6, CLS=7, SEP=8, MASK=9.
# GPT-2 causal LM doesn't really use bos/eos in our chunks (each row is a
# fixed 1024-nt block), but we expose them for parity with the legacy config.
bos_token_id = tokenizer.cls_token_id  # 7
eos_token_id = tokenizer.sep_token_id  # 8
pad_token_id = tokenizer.pad_token_id  # 5
