# Resume genome_sequence × gpt2-small from iter 50000 with extended training and
# a constant learning rate. Used for the number_sample bugfix follow-up where the
# original 50k-iter / max_iters=50000 schedule turned out to be insufficient
# compute for the larger (500k-cap = 3 B token) corpus produced after the cap fix
# — see docs/_tmp/20260520-number-sample-impact-retrain-list.md §9.7.x.
#
# Imports + dataset_dir / out_dir / tokenizer setup are identical to gpt2_small.py;
# only the iter budget, the LR schedule, and the checkpoint-keep flag change.

import sentencepiece as spm

from molcrawl.core.paths import (
    REFSEQ_DATASET_DIR,
    get_gpt2_output_path,
    get_refseq_tokenizer_path,
)

tokenizer_path = get_refseq_tokenizer_path()
dataset_dir = REFSEQ_DATASET_DIR

tensorboard = True
tensorboard_dir = get_gpt2_output_path("genome_sequence", "small")
out_dir = get_gpt2_output_path("genome_sequence", "small")

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
meta_vocab_size = tokenizer.vocab_size()

# Same global batch as gpt2_small.py — 12 * 1024 * 5 * 8 = 491,520 tokens/iter.
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# === Extension knobs ===
init_from = "resume"        # pick up ckpt.pt at out_dir/iter 50000

# 3x the previous budget. The previous run consumed 24.5 B tokens and stopped
# with val_loss=6.23 / PPL≈506. OLD (50k-cap) at 89k iter / 43.7 B tokens
# achieved PPL≈295 — i.e. its 1.78x larger compute is what beat us, not its
# data. Catching up needs ~90k iter at minimum; pad to 150k so we cleanly
# surpass OLD compute and have room to observe the data-amount advantage.
max_iters = 150000
lr_decay_iters = 150000     # unused when decay_lr=False but kept consistent

# Constant LR. Resuming with the original cosine schedule would compute
# get_lr(50000) under the new lr_decay_iters=150000 and re-inflate the LR
# from min_lr (where the 50k run finished) back to roughly 4e-6 — an
# unintended re-warmup that can destabilise an already-converged model.
# A constant LR halfway between the previous max (6e-6) and min (6e-7)
# avoids that jump while still letting the model continue improving.
decay_lr = False
learning_rate = 3e-6
min_lr = learning_rate / 10  # ignored when decay_lr=False, kept for API symmetry
warmup_iters = 0             # already warmed up in the original run

# eval stuff (unchanged)
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Checkpoint policy
save_hf_checkpoints = True
save_checkpoint_steps = 5000
max_checkpoints = 10
keep_legacy_ckpt = True       # preserve ckpt.pt so existing Tier A/B eval scripts continue to work
always_save_checkpoint = False

# Early stopping — relax compared to gpt2_small.py so the longer run can ride
# out plateaus that the original short run hit early. Patience 10 = 10000 iter
# (1 eval = 1000 iter).
early_stopping = True
early_stopping_patience = 10

# Regularisation: keep dropout consistent with the original small run.
weight_decay = 1e-1
dropout = 0.1
