# compounds GPT-2 medium — 8e-4, seed 42, fixed evaluation set
# launch: torchrun --standalone --nproc_per_node=4 molcrawl/models/gpt2/train.py <this config>
#
# A rerun of this size's best rate under the evaluation the large and xl arms already
# use. Read off the manifests on 2026-10-14, the four best arms differed in two ways
# at once:
#
#   size    rate     seed  deterministic_val_eval  eval_sequences_seen
#   small   8e-4     1001  (absent, so False)      3,200
#   medium  8e-4     1001  (absent, so False)      3,200
#   large   1.6e-3   42    True                    3,200
#   xl      1.6e-3   42    True                    3,200
#
# All four average over 3,200 val sequences, but small and medium redraw those 3,200
# with replacement at every eval while large and xl score one fixed set. A minimum
# over 95 noisy eval points sits below the same minimum over 95 quiet ones, so the
# small and medium figures carry a downward bias the large and xl figures do not, and
# the ladder's apparent turn after medium is partly that bias. Seed moves with it:
# 1001 against 42. This config removes both differences at once, which is the only
# way to remove either -- rerunning at 1001 would leave the seed mismatched, and
# switching seed without fixing the eval would leave the bias.
#
# large and xl are not rerun. They already hold seed 42, a fixed eval set and 3,200
# sequences, so a rerun would reproduce them.
#
# The membership of the fixed set comes from a constant (12345 in train.py), not from
# the run's seed, so these two and the existing large and xl arms all score the same
# val sequences.
#
# seed 42 per the standing directive of 2026-09-30. data_seed is not written: neither
# path reads it and both refuse a config that declares one different from seed.

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_GPT2, get_gpt2_output_path

# Medium-Sized GPT2 Model

n_layer = 24
n_head = 16
n_embd = 1024

dataset_dir = COMPOUNDS_DATASET_DIR_GPT2

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = get_gpt2_output_path("compounds", "medium") + "-lr8e4-seed42"
out_dir = get_gpt2_output_path("compounds", "medium") + "-lr8e4-seed42"

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)
meta_vocab_size = tokenizer.vocab_size
eos_token_id = tokenizer.eos_token_id  # 13 ([SEP]) — the molecule separator in packed blocks

# nanoGPT divides gradient_accumulation_steps by the DDP world size, so the
# effective global batch = batch_size * gradient_accumulation_steps and is
# GPU-count-independent. 16 * 160 = 2560 seq, as the rest of this ladder.
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 160  # 16 * 160 = 2560 seq global batch

# train.py refuses to start when micro x accumulation disagrees with this.
expected_global_batch = 2560

# v4 packed data (2026-08-05): train = 398,917 blocks x 1024, no padding.
# 4,674 iters at global batch 2,560 = 29.99 epochs, the length the existing sweep
# points and the large and xl arms all ran at.
max_iters = 4674
lr_decay_iters = 4674
warmup_iters = 93  # 2% of max_iters
learning_rate = 0.0008  # this size's best rate in the existing sweep
min_lr = 8e-05  # ~= learning_rate/10 per Chinchilla

# eval stuff — 95 eval points, as every arm this will be tabulated against.
eval_interval = 50
# Fix the number of val *sequences* rather than batches, so sizes whose batch_size
# differs still average over the same sample. train.py derives eval_iters from this
# and batch_size, so setting eval_iters here too would be dead config.
eval_sequences = 3200
log_interval = 10

init_from = "scratch"  # v4 packed ladder starts fresh; no v3 checkpoint is compatible

# Checkpoint policy, unified across modalities and architectures (directive
# 2026-09-15 §3.1).
always_save_checkpoint = False
save_checkpoint_steps = 1000
max_checkpoints = 10

# early stopping — OFF for pretraining: the ladder is compute-matched and every size
# runs the full schedule.
early_stopping = False

weight_decay = 0.1

dataset = "compounds"

# Special Tokens
start_instruction = 12
eos_token = 12  # eos

dataset_params = {"dataset_dir": dataset_dir}

seed = 42

# Score the same fixed val sequences at every evaluation instead of resampling with
# replacement. This is the difference being removed.
deterministic_val_eval = True
