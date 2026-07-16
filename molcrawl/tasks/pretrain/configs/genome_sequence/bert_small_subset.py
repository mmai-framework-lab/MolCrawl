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
# LR sweep on the pre-G2 (2026-05) mammal_centered dataset (jobs 19018 +
# 19043-19045) showed 1e-4 collapsed to loss ≈ ln(4) ≈ 1.39 (degenerate
# near-uniform output) and 1e-5 was the only value that trained. That result
# is documented here for reference but does NOT drive the current shipped
# value: the boss's 2026-07-14 reply mandates BERT LR = 1e-4 unified across
# every modality/size, and the G2 dataset (contig-split, chr22 hold-out,
# 83M-window budget) is not identical to the old flow that collapsed. The
# readiness report flags this specifically so the LR call can be revisited
# before the full 42-config run launches — do not use this file's default
# blind for another sweep without checking the report.
# BERT_LR_TAG env var stays as the escape hatch for sweeps (checkpoint dir
# is suffixed so concurrent LR values do not overwrite each other).
learning_rate = float(os.environ.get("SUBSET_BERT_LR", "0.0001"))
weight_decay = 0.01
# Fixed-schedule comparison run (charter §「比較系は early_stopping OFF、
# compute-matched」). Overriding max_steps / warmup_steps below.
early_stopping = False

# Compute-matched schedule from the realized dataset row count (charter
# Condition 2 output): global batch 2,560 × 3 epochs per subset. Reading
# via `load_from_disk` is memory-mapped, so this only touches metadata.
_GLOBAL_BATCH = 2560
_N_EPOCH = 3
from datasets import load_from_disk as _load
_ds_for_len = _load(dataset_dir)
_train_n = len(_ds_for_len["train"])
max_steps = (_N_EPOCH * _train_n + _GLOBAL_BATCH - 1) // _GLOBAL_BATCH
warmup_steps = max(int(0.02 * max_steps), 100)  # ≈ 2 % of max_steps
del _ds_for_len

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
