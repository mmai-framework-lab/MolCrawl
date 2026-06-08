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
_subset_suffix = f"-{GENOME_SUBSET}"
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
batch_size = 12
block_size = 1024  # matches Phase 3 gpt2_chunk_size
gradient_accumulation_steps = 5 * 8

max_iters = 50000
lr_decay_iters = 50000
# GPT-2 pretrain learning rate per Radford et al. (2019) for the 124M model:
# 6e-4 peak with linear warmup of ~2,000 iters, cosine decay to 10% of peak.
# Legacy genome GPT-2 configs all use 6e-6, but that value is undocumented
# (initial commit 2025-03-25, never revisited) and not justified by any
# divergence experience in the git history. We use the literature value.
warmup_iters = 2000
learning_rate = 6e-4
min_lr = learning_rate / 10  # → 6e-5 (10% of peak per Chinchilla / GPT-2 convention)

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

weight_decay = 1e-1

dataset = "genome_sequence"
dataset_params = {"dataset_dir": dataset_dir}

# Special token ids (single-nucleotide vocab from single_nuc_tokenizer):
#   PAD=5, UNK=6, CLS=7, SEP=8, MASK=9.
# GPT-2 causal LM doesn't really use bos/eos in our chunks (each row is a
# fixed 1024-nt block), but we expose them for parity with the legacy config.
bos_token_id = tokenizer.cls_token_id  # 7
eos_token_id = tokenizer.sep_token_id  # 8
pad_token_id = tokenizer.pad_token_id  # 5
