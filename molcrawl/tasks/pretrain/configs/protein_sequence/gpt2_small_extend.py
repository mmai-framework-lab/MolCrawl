# Resume protein_sequence × gpt2-small from iter 41000 with a constant LR.
# Same pattern as genome_sequence/gpt2_small_extend.py — see that file for
# the rationale (avoid cosine re-warmup, give NEW compute parity with the
# legacy 50k-cap run).
#
# protein gpt2-small terminated by early_stop after 11h50m at step 41000
# (val 2.5885, best ~step 36000?). The trajectory was still improving when
# patience expired — extending with relaxed patience + constant mid-range LR
# should let it ride past the plateau if there is still signal.

from molcrawl.core.paths import UNIPROT_DATASET_DIR, get_gpt2_output_path
from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer as Tokenizer

dataset_dir = UNIPROT_DATASET_DIR

tensorboard_dir = get_gpt2_output_path("protein_sequence", "small")
out_dir = get_gpt2_output_path("protein_sequence", "small")

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# Extension
init_from = "resume"
max_iters = 150000
lr_decay_iters = 150000
decay_lr = False
learning_rate = 1e-4          # constant LR midway between 6e-4 and 6e-5
min_lr = learning_rate / 10
warmup_iters = 0

# Eval / checkpoint cadence
eval_interval = 1000
eval_iters = 200
log_interval = 10

save_hf_checkpoints = True
save_checkpoint_steps = 5000
max_checkpoints = 10
keep_legacy_ckpt = True
always_save_checkpoint = False

# Relaxed early stopping (was patience=5)
early_stopping = True
early_stopping_patience = 10

# Regularisation
weight_decay = 1e-1
dropout = 0.1

# Dataset wiring (mirrors gpt2_small.py)
dataset = "protein_sequence"
dataset_params = {"dataset_dir": dataset_dir}

# HF token IDs for ESM tokenizer
bos_token_id = 0
eos_token_id = 2
pad_token_id = 1
