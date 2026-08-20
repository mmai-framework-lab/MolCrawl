# compounds GPT-2 xl — packed 1024 ladder (v4 data, 2026-08-05)
# launch: torchrun --standalone --nproc_per_node=4 molcrawl/models/gpt2/train.py <this config>


from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_GPT2, get_gpt2_output_path

# EX-Large-Sized GPT2 Model
n_layer = 48
n_head = 25
n_embd = 1600

dataset_dir = COMPOUNDS_DATASET_DIR_GPT2

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = get_gpt2_output_path("compounds", "xl")
out_dir = get_gpt2_output_path("compounds", "xl")

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)
meta_vocab_size = tokenizer.vocab_size
eos_token_id = tokenizer.eos_token_id  # 13 ([SEP]) — the molecule separator in packed blocks

# nanoGPT divides gradient_accumulation_steps by the DDP world size, so the
# effective global batch = batch_size * gradient_accumulation_steps and is
# GPU-count-independent. 4 * 640 = 2560 seq (same convention as protein xl).
# The previous "batch x grad_accum x n_GPUs(4) = 2,560" comment was wrong for
# nanoGPT semantics: 2 * 320 was an effective 640, not 2,560.
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 640  # 4 * 640 = 2560 seq global batch

# v4 packed data (2026-08-05): train = 398,917 blocks x 1024, no padding.
# 10 epochs at global batch 2560 = floor(10 * 398,917 / 2560) = 1,558 iters.
max_iters = 1558
lr_decay_iters = 1558
warmup_iters = 31  # ~2% of max_iters; must stay < max_iters so LR reaches peak
learning_rate = 0.0002  # max learning rate
min_lr = 2e-05  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff — ~31 eval points over the run; log_interval != eval_interval so the
# reported dt/MFU is not polluted by eval time.
eval_interval = 50
eval_iters = 200
# eval_sequences fixes the *number of validation sequences* per eval point instead of
# the number of batches, so every ladder size averages its val loss over the same
# 3,200 sequences. batch_size shrinks with model size, so the old shared
# eval_iters=200 gave the large models a 2-4x smaller val sample.
eval_sequences = 3200
log_interval = 10

# init from checkpoint
init_from = "scratch"  # v4 packed ladder starts fresh; no v3 checkpoint is compatible

# checkpoint management
always_save_checkpoint = True  # Save regularly regardless of validation loss
save_checkpoint_steps = None  # If None, save with eval_interval
max_checkpoints = 5  # Keep up to 5 checkpoints

# early stopping — OFF for pretraining: the ladder is compute-matched, every size
# runs the full schedule (spec 2026-07-31 §2; matches bert_*.py early_stopping=False).
early_stopping = False

# weight decay
weight_decay = 0.1

# dataset
dataset = "compounds"

# Special Tokens
start_instruction = 12
eos_token = 12  # eos

dataset_params = {"dataset_dir": dataset_dir}

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 22
