# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


from molcrawl.core.paths import UNIPROT_DATASET_DIR, get_gpt2_output_path
from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer as Tokenizer

# Large-Sized GPT2 Model

n_layer = 36
n_head = 20
n_embd = 1280

dataset_dir = UNIPROT_DATASET_DIR

tensorboard_dir = get_gpt2_output_path("protein_sequence", "large")
out_dir = get_gpt2_output_path("protein_sequence", "large")

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size

# Effective global batch = 2560 sequences (spec; protein GPT-2 applied first, other modalities/archs to follow;
# see tmp/protein-global-batch-analysis-2026-08-04.md). GPT-2 effective batch =
# batch_size * gradient_accumulation_steps, GPU-count-independent. 8 * 320 = 2560
# (smaller micro-batch than small/medium for GPU memory on the larger model).
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 320  # 8 * 320 = 2560 seq global batch

# 3 epochs of the train split at global batch 2560:
# floor(3 * 9,538,464 train blocks / 2560) = 11,177 iters (~29.3B tokens processed).
max_iters = 11177
lr_decay_iters = 11177
warmup_iters = 224  # ~2% of max_iters (compounds convention); < max_iters so LR reaches peak

# eval stuff
eval_interval = 1000
# eval_sequences fixes the *number of validation sequences* per eval point instead of
# the number of batches, so every ladder size averages its val loss over the same
# 3,200 sequences. batch_size shrinks with model size, so a shared eval_iters would
# give the large models a 2-4x smaller val sample. train.py derives eval_iters from
# this and batch_size, so setting eval_iters here as well would be dead config.
eval_sequences = 3200
log_interval = 10

# init from checkpoint
init_from = "resume"  # 'scratch' or 'resume' - resume from checkpoint by default

# checkpoint management
always_save_checkpoint = True  # Save regularly regardless of validation loss
save_checkpoint_steps = None  # If None, save with eval_interval
max_checkpoints = 5  # Keep up to 5 checkpoints

# early stopping
early_stopping = True
early_stopping_patience = 10  # increased from 5 to allow more exploration with dropout

# learning rate (increased from 6e-6 to compensate for dropout regularisation)
learning_rate = 0.00025
min_lr = 2.5e-05

# regularisation
weight_decay = 0.1
dropout = 0.1

# dataset
dataset = "protein_sequence"

dataset_params = {
    "dataset_dir": dataset_dir  # Adjust the path as necessary for your generated dataset.
}

# --- MolCrawl HF token IDs (added by patch_configs.py) ---
# EsmSequenceTokenizer: <cls>=0, <pad>=1, <eos>=2
bos_token_id = 0
eos_token_id = 2
pad_token_id = 1

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 84
