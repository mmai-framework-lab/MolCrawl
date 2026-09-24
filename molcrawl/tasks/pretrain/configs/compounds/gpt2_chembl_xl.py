# GPT-2 (xl) fine-tuning config for ChEMBL
#
# Continues from the compounds GPT-2 pretraining checkpoint
# (see molcrawl/tasks/pretrain/configs/compounds/gpt2_xl.py)
# using the ChEMBL fine-tuning dataset.
#
# Recommended launch command:
#   torchrun --standalone --nproc_per_node=<N> molcrawl/models/gpt2/train.py \
#       gpt2/configs/compounds/train_gpt2_chembl_xl.py

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import CHEMBL_DATASET_DIR, get_gpt2_output_path

# EX-Large-Sized GPT-2 Model
n_layer = 48
n_head = 25
n_embd = 1600

tensorboard = True
tensorboard_dir = get_gpt2_output_path("compounds_chembl", "xl")
out_dir = get_gpt2_output_path("compounds_chembl", "xl")
# Pretraining checkpoint to load weights from when out_dir has no checkpoint.
pretrain_dir = get_gpt2_output_path("compounds", "xl")

tokenizer_path = "assets/molecules/vocab.txt"
tokenizer = Tokenizer(tokenizer_path, 256)
meta_vocab_size = tokenizer.vocab_size
eos_token_id = tokenizer.eos_token_id  # 13 ([SEP])

dataset_dir = CHEMBL_DATASET_DIR

# Batch / block settings — same as pretraining
batch_size = 2
block_size = 1024
gradient_accumulation_steps = 5 * 16
# The effective global batch this config runs at. nanoGPT divides the accumulation
# by the DDP world size and multiplies it back, so micro x accumulation is the
# effective batch whatever the GPU count, and train.py refuses to start when the
# two disagree.
#
# Declared at 160, which is what this config computes -- not at the 2,560 the
# organix13 ladder uses. The chembl subsets were set up with their own shape and
# no run of theirs at 2,560 exists; writing 2,560 here would stop the run rather
# than describe it. Any move to 2,560 is a change to the run, not to a comment.
expected_global_batch = 160


# Fine-tuning schedule: fewer iterations and a lower LR than pretraining
# (pretraining: max_iters=30000, lr=6e-7)
max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 100
learning_rate = 1e-5
min_lr = learning_rate / 10

# Evaluation
eval_interval = 200
eval_iters = 200
log_interval = 50

# Resume from compounds pretraining checkpoint
init_from = "resume"

# Checkpoint management
always_save_checkpoint = True
save_checkpoint_steps = None
max_checkpoints = 5

# early stopping
early_stopping = True
early_stopping_patience = 5

# Regularisation
weight_decay = 1e-1

# Dataset identifier used by the data-loader
dataset = "compounds_chembl"

# Special Tokens (SMILES tokenizer: [CLS]=2, [SEP]=3)
start_instruction = 2
eos_token = 2

dataset_params = {
    "dataset_dir": dataset_dir,
}

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 14
