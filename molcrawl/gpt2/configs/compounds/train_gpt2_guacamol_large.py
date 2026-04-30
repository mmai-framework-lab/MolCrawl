# GPT-2 (large) fine-tuning config for GuacaMol benchmark
#
# Continues from the compounds GPT-2 pretraining checkpoint
# (see molcrawl/gpt2/configs/compounds/train_gpt2_large_config.py)
# using the GuacaMol benchmark SMILES dataset.
#
# Recommended launch command:
#   torchrun --standalone --nproc_per_node=<N> molcrawl/gpt2/train.py \
#       gpt2/configs/compounds/train_gpt2_guacamol_large.py

from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import GUACAMOL_DATASET_DIR, get_gpt2_output_path

# Large-Sized GPT-2 Model
n_layer = 36
n_head = 20
n_embd = 1280

tensorboard = True
tensorboard_dir = get_gpt2_output_path("compounds_guacamol", "large")
out_dir = get_gpt2_output_path("compounds_guacamol", "large")
# Pretraining checkpoint to load weights from when out_dir has no checkpoint.
pretrain_dir = get_gpt2_output_path("compounds", "large")

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)
meta_vocab_size = tokenizer.vocab_size
eos_token_id = tokenizer.eos_token_id  # 13 ([SEP])

dataset_dir = GUACAMOL_DATASET_DIR

# Batch / block settings — same as pretraining
batch_size = 2  # max size in koala
block_size = 1024
gradient_accumulation_steps = 5 * 16

# Fine-tuning schedule: fewer iterations and a lower LR than pretraining
# (pretraining: max_iters=30000, lr=6e-7)
max_iters = 2000
lr_decay_iters = 2000
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
dataset = "compounds"

# Special Tokens
start_instruction = 12
eos_token = 12

dataset_params = {
    "dataset_dir": dataset_dir,
}
