"""Genome BERT (small) — single-nucleotide / subset (Evo2 species) flow.

Reference config for training BERT on the Phase 3 parquet produced by the
subset pipeline (``parquet_bert/<accession>.parquet``, 512-token rows with
[CLS]+510+[SEP] and an all-ones attention_mask, NO baked MLM masks).

Differences from :mod:`bert_small`:
- Vocab is 10 (A/T/G/C/N + PAD/UNK/CLS/SEP/MASK), not 4096 BPE.
- Tokenizer is a character-level HF tokenizer built from
  :mod:`molcrawl.data.genome_sequence.utils.single_nuc_tokenizer`.
- ``dataset_dir`` points at the subset's ``training_ready_hf_dataset_bert/``
  Arrow DatasetDict (Phase 6 output); the subset name comes from the
  ``GENOME_SUBSET`` env var so workflows can switch subsets without
  editing this file. The genome BERT trainer (``molcrawl/models/bert/main.py``)
  consumes this via ``load_from_disk`` and indexes as ``dataset["train"]``
  / ``dataset["test"]`` — pointing at the raw ``parquet_bert/`` directory
  would crash at startup since that path is not a DatasetDict.
- ``out_dir`` / ``tensorboard_dir`` are suffixed with the subset name so each
  subset run gets its own checkpoint tree.
- Dynamic MLM masking via ``DataCollatorForLanguageModeling`` (unchanged from
  the legacy bert/main.py path; the parquet deliberately ships no labels).
"""

import os

from transformers import AutoTokenizer

from molcrawl.core.paths import (
    GENOME_SEQUENCE_DIR,
    get_bert_output_path,
    get_custom_tokenizer_path,
)
from molcrawl.data.genome_sequence.utils.single_nuc_tokenizer import (
    build_single_nuc_tokenizer,
)

# ---- subset selection ----------------------------------------------------- #
GENOME_SUBSET = os.environ.get("GENOME_SUBSET")
if not GENOME_SUBSET:
    raise RuntimeError(
        "GENOME_SUBSET env var is required for the subset BERT config. "
        "Example: GENOME_SUBSET=mammal_centered python -m molcrawl.models.bert.main ..."
    )

# ---- paths ---------------------------------------------------------------- #
# BERT_LR_TAG (optional): appended to the checkpoint dir so concurrent LR
# sweeps for the same subset land in distinct trees (e.g. "lr1e-5", "lr3e-5").
# Leave unset for production runs so the canonical subset path is used.
model_size = "small"
_lr_tag = os.environ.get("BERT_LR_TAG", "")
_subset_suffix = f"-{GENOME_SUBSET}" + (f"-{_lr_tag}" if _lr_tag else "")
model_path = get_bert_output_path("genome_sequence", model_size) + _subset_suffix
dataset_dir = f"{GENOME_SEQUENCE_DIR}/{GENOME_SUBSET}/training_ready_hf_dataset_bert"

# ---- tokenizer (10-symbol single-nucleotide, shared across subsets) ------ #
_custom_tokenizer_path = get_custom_tokenizer_path("genome_sequence", "bert_single_nuc")
build_single_nuc_tokenizer(_custom_tokenizer_path)  # idempotent build/cache
tokenizer = AutoTokenizer.from_pretrained(_custom_tokenizer_path)
vocab_size = len(tokenizer)  # = 10

# ---- training hyperparameters (mirror bert_small) ------------------------ #
max_length = 512  # = [CLS] + 510 nucleotides + [SEP] (matches Phase 3 chunking)
# LR sweep over {1e-4, 1e-5, 3e-5, 5e-5} on mammal_centered (jobs 19018 +
# 19043 / 19044 / 19045) showed:
#   1e-4 → collapses to loss ≈ ln(4) ≈ 1.39 (degenerate near-uniform output)
#   5e-5 → plateaus around 1.263 (near-degenerate, no learning signal)
#   3e-5 → microscopic improvement (Δ=0.006 over 15k iters)
#   1e-5 → consistent descent 1.56 → 1.256 (Δ=0.31, lowest grad_norm)
# So 1e-5 is the only value that actually trains this small × vocab=10 × bf16
# configuration. The Devlin et al. recipe (1e-4 × 10k warmup) targets BERT-base
# (110M × vocab 30k × NLP) and does not transfer. warmup_steps stays at 10k
# to follow the literature ratio (warmup ≈ max_steps/6). Override via env for
# future sweeps (use BERT_LR_TAG to keep checkpoint trees distinct).
learning_rate = float(os.environ.get("BERT_LR", "1e-5"))
warmup_steps = int(os.environ.get("BERT_WARMUP_STEPS", "10000"))
max_steps = int(os.environ.get("BERT_MAX_STEPS", "60000"))
weight_decay = 1e-1
early_stopping = False

log_interval = 100
save_steps = 1000

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16

# ---- performance opt-ins (consumed by bert/main.py via globals().get) ----- #
# bf16 = bfloat16 mixed precision (Hopper/Blackwell — RIKEN H100/H200 OK).
# Together with the dataloader flags this roughly halves per-step wallclock
# on H100/H200 vs the legacy fp32 + 0-worker dataloader defaults.
bf16 = True
dataloader_num_workers = 4
dataloader_pin_memory = True

# Deliberately no ``preprocess_function``: the Phase 6 Arrow DatasetDict
# already ships ``input_ids`` + ``attention_mask`` for every row, so defining
# a config-level ``preprocess_function`` would push ``bert/main.py`` into its
# ``elif "preprocess_function" in globals():`` branch and trigger a full
# ``.map()`` over ~95M rows just to re-derive a mask that is already correct.
