# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


from molcrawl.core.paths import UNIPROT_DATASET_DIR, get_gpt2_output_path
from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer as Tokenizer

# Medium-Sized GPT2 Model

n_layer = 24
n_head = 16
n_embd = 1024

dataset_dir = UNIPROT_DATASET_DIR

tensorboard_dir = get_gpt2_output_path("protein_sequence", "medium")
out_dir = get_gpt2_output_path("protein_sequence", "medium")

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size

# Effective global batch = 2560 sequences (spec; protein GPT-2 applied first, other modalities/archs to follow;
# see tmp/protein-global-batch-analysis-2026-08-04.md). GPT-2 effective batch =
# batch_size * gradient_accumulation_steps, GPU-count-independent. 16 * 160 = 2560.
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 160  # 16 * 160 = 2560 seq global batch

# 3 epochs of the train split at global batch 2560:
# floor(3 * 9,538,464 train blocks / 2560) = 11,177 iters (~29.3B tokens processed).
max_iters = 33531        # 9 epochs of the current packed train: floor(9*9,538,464/2560). No data rebuild (rev2 §1).
expected_global_batch = 2560  # startup guard: fail if batch_size*grad_accum*world_size != 2560  # retrain rev2 2026-09-07
lr_decay_iters = 33531   # = max_iters
warmup_iters = 671       # 2% of max_iters

# eval stuff
eval_interval = 335      # ~100 eval points over max_iters
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
save_checkpoint_steps = 1000  # resumable latest every 1000 steps (4-day TimeLimit + requeue resume)  # retrain rev2 2026-09-07
max_checkpoints = 5  # Keep up to 5 checkpoints

# early stopping
early_stopping = False   # no patience: run to completion  # retrain 2026-09-07
early_stopping_patience = 10  # increased from 5 to allow more exploration with dropout

# learning rate (increased from 6e-6 to compensate for dropout regularisation)
learning_rate = 6e-4     # template default (= L, arm base). The 21-run retrain overrides --learning_rate per arm at launch (rev2 §3: L, L/2, L/4; xl also L/8).
min_lr = 6e-5            # template (peak/10). Overridden per arm at launch (= arm LR / 10).
dtype = "bfloat16"       # explicit precision, unified across sizes  # retrain 2026-09-07

# regularisation
weight_decay = 0.1
dropout = 0.0            # 0 for clean size comparison (was 0.1)  # retrain 2026-09-07

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
seed = 42                # template default. Overridden per run at launch (rev2 §3 run seed: 1001/1002/1003).
