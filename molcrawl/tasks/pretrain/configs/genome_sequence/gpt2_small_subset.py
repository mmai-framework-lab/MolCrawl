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

from datasets import load_from_disk as _load
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
# Effective global batch = 2560 sequences (spec, unified across modalities/archs;
# batch_size * gradient_accumulation_steps, GPU-count-independent because
# train.py does `grad_accum //= world_size`). Matches the protein GPT-2 fix.
#
# The previous 8 × (5 * 16) = 8 × 80 was written as if grad_accum were
# per-device, and the stale comment claimed "per_device 8 × grad_accum 80 ×
# n_GPUs 4 = 2,560". Under nanoGPT semantics that yielded an effective global
# batch of 640 (measured: "tokens per iteration will be: 655,360" = 640 × 1024),
# i.e. 4x below spec, which left GPT-2 at 0.75 epoch while BERT (HF Trainer,
# per-device semantics) ran the intended 3 epochs — breaking the compute-matched
# cross-arch comparison the 42-run charter is built on.
#
# max_iters below already assumes _GLOBAL_BATCH = 2560, so it needs no change.
# The 8/320 split is an operational choice, not the spec: only the product is
# fixed. protein already varies it by size (small 8 × 320, medium 16 × 160). A
# larger micro-batch shortens wall-clock at the same effective batch, which
# matters where a scheduler caps job duration, so re-probe throughput and adjust
# the split before launching production.
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 320  # 8 * 320 = 2560 seq global batch

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

# What the schedule below was derived from, stated so train.py can check it.
# nanoGPT's effective batch does not move with the GPU count, so the failure it
# guards against is quieter: batch_size and gradient_accumulation_steps get
# edited apart and their product stops matching. This config trained at 640 for
# exactly that reason before 4974f55.
expected_global_batch = _GLOBAL_BATCH
# Epochs. The default is the 3 the production 21 ran at; SUBSET_GPT2_EPOCHS
# raises it without editing this line, the same way SUBSET_BERT_EPOCHS does on
# the BERT side. It has to go through max_iters rather than through a
# command-line --max_iters, because warmup_iters and lr_decay_iters are derived
# from it here: overriding only max_iters at launch would leave a 9-epoch run
# warming up over 2% of a 3-epoch schedule, which is 0.67% of its own.
_N_EPOCH = int(os.environ.get("SUBSET_GPT2_EPOCHS", "3"))
# SUBSET_GPT2_LR_DECAY_EPOCHS detaches the cosine schedule's length from the
# run's length. Unset it and the two stay equal, which is every run so far.
#
# It exists for the schedule measurement the 2026-10-15 judgment asks for. The
# 9-epoch campaign tripled max_iters, and because lr_decay_iters followed it the
# stretch spent above 5e-5 tripled too: 29,215 steps became 87,643. Only 6 of 21
# runs improved and 20 of 21 reversed by more than 0.01, so the question is
# whether the extra data or the longer high-LR stretch did that. Setting this to
# the old epoch count holds the whole LR curve -- warmup, cosine, floor -- where
# the 3-epoch runs had it and lets the run continue at min_lr past its end, which
# separates the two.
_N_EPOCH_DECAY = int(os.environ.get("SUBSET_GPT2_LR_DECAY_EPOCHS", str(_N_EPOCH)))
_ds_for_len = _load(dataset_dir)
_train_n = len(_ds_for_len["train"])


def _iters_for(n_epoch):
    return (n_epoch * _train_n + _GLOBAL_BATCH - 1) // _GLOBAL_BATCH


max_iters = _iters_for(_N_EPOCH)
lr_decay_iters = _iters_for(_N_EPOCH_DECAY)
# Warmup follows the decay length, not the run length, so the curve's leading
# edge is identical to the schedule being borrowed. Deriving it from max_iters
# instead would warm a 9-epoch run over 3,278 steps against the 1,092 the
# 3-epoch schedule used, which is a second difference inside a measurement of
# one.
warmup_iters = max(int(0.02 * lr_decay_iters), 100)  # ≈ 2 % of lr_decay_iters
del _ds_for_len

# Smoke-test override for readiness verification only (charter 2026-07-14).
# `SMOKE_MAX_STEPS=5` forces a short run that exercises the training path
# without consuming a real epoch. Untouched in production runs.
_smoke = os.environ.get("SMOKE_MAX_STEPS")
if _smoke:
    max_iters = int(_smoke)
    lr_decay_iters = max_iters
    warmup_iters = max(int(0.4 * max_iters), 1)
    # Also shrink eval_interval so a short smoke actually captures several
    # eval samples (the production 1000 leaves an 800-step smoke with 0-1
    # evals, breaking downstream aggregation).
    eval_interval = max(int(max_iters / 5), 50)
    # SMOKE_WARMUP_ITERS / SMOKE_EVAL_INTERVAL (opt) override the derived
    # values — used by the 2026-07-15 extended sweep (8000 iter) so we get
    # production-realistic warmup (~2%) and finer-grained evals for spike
    # detection around the pre-G2 6k-iter warning point.
    _smoke_warmup = os.environ.get("SMOKE_WARMUP_ITERS")
    if _smoke_warmup:
        warmup_iters = int(_smoke_warmup)
    _smoke_eval = os.environ.get("SMOKE_EVAL_INTERVAL")
    if _smoke_eval:
        eval_interval = int(_smoke_eval)

# Fixed-schedule comparison run (charter §: comparison runs use
# early_stopping OFF and are compute-matched).  gpt2/train.py reads this via globals().get.
early_stopping = False

# The validation set is fixed from 2026-10-22. The judgment of that date
# withdraws the earlier one that let GPT-2 redraw, so each evaluation now reads
# the same 64,000 sequences instead of drawing 64,000 with replacement. The
# count does not change: eval_iters 200 x batch_size 320 either way.
#
# What this removes is the downward bias on best_val. Redrawing made every
# evaluation a noisy estimate, and taking the minimum over 41-55 of them (three
# epochs) or 124-165 (nine) picked up a different amount of that noise at each
# length, so the two series' best values were not on one footing.
deterministic_val_eval = True

# eval_interval for production runs. 1,000 was the charter's compute-matched
# value; 100 is the cross-modality value (directive 2026-09-15 §3). An eval is
# eval_iters x 2 splits x batch_size forwards on rank 0 alone, about 1.7 training
# steps here, so ten times as many costs under 2 % of the run. The 16.3 % figure
# that widened genome's interval was BERT's, and came from evaluating at a
# micro-batch 20x below the training one -- a different problem, fixed separately.
# When SMOKE_MAX_STEPS is set the block above already overrode this, so guard.
if not _smoke:
    eval_interval = 100
eval_iters = 200
log_interval = 10

# "scratch" for first pretraining run; flip to "resume" on subsequent
# resumes from out_dir/<checkpoint>/. Previously this was hard-coded to
# "resume" (mirroring the legacy gpt2_small.py), which crashes the trainer
# on a fresh subset run because no checkpoint exists yet.
init_from = "scratch"
# Checkpoint policy, unified across modalities and architectures (directive
# 2026-09-15 §3.1). always_save_checkpoint writes at every eval point, which makes
# save_checkpoint_steps mean nothing; off, the run writes on the periodic grid for
# resume and on every improvement for comparison (train.py, is_best_model).
always_save_checkpoint = False
save_checkpoint_steps = 1000
# Phase 7 (2026-07-22) added the env override so LR-fix pilots could retain more
# ckpts: the best-val one lands at step 5k-13k for a divergent run and the 50k+
# schedule evicted it. cleanup_old_checkpoints now keeps the best by score rather
# than the newest, so the window no longer decides that; the override stays as an
# escape hatch. Default raised 5 -> 10 to match the other modalities.
max_checkpoints = int(os.environ.get("SUBSET_GPT2_MAX_CKPT", "10"))

# Phase 7 (2026-07-22): env override for weight_decay to support LR-fix pilot
# where BERT-equivalent wd=0.01 is being tested (per boss GO in
# reply-gpt2-lr-fix-and-seed10-smoke.md §A). Default 0.1 preserved (charter §3.5).
weight_decay = float(os.environ.get("SUBSET_GPT2_WD", "0.1"))

dataset = "genome_sequence"
dataset_params = {"dataset_dir": dataset_dir}

# Special token ids (single-nucleotide vocab from single_nuc_tokenizer):
#   PAD=5, UNK=6, CLS=7, SEP=8, MASK=9.
# GPT-2 causal LM doesn't really use bos/eos in our chunks (each row is a
# fixed 1024-nt block), but we expose them for parity with the legacy config.
bos_token_id = tokenizer.cls_token_id  # 7
eos_token_id = tokenizer.sep_token_id  # 8
pad_token_id = tokenizer.pad_token_id  # 5

# Phase 0-1 pad-loss-mask hook: mask CLM loss at pad-token positions so the
# subset GPT-2 does not waste capacity learning "predict PAD after PAD" in
# short-genome-window tails. Value must match `pad_token_id` (=5, single-nt
# vocab). Without this the receiving `train.py` code (commit 17562e81) sees
# `pad_token_id_for_loss = None` and skips the mask entirely.
pad_token_id_for_loss = 5

# Training seed. 45 until 2026-10-22, from the per-config assignment of
# 2026-08-03; the judgment of 2026-10-22 puts every genome run on 42, BERT and
# GPT-2 alike, and asks that no other value be used and that seeds not be
# varied to measure spread. The runs already on disk keep the seed they ran
# with and are not re-seeded; they are re-run instead.
#
# data_seed is written because the same judgment asks for it. It does nothing
# on this path -- train.py has no data_seed, and one seed drives both the
# weights and get_batch's draw -- and the guard accepts it only because it
# equals seed. It is here so the config says what it was asked to say.
seed = 42
data_seed = 42
