# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


from molcrawl.config.paths import UNIPROT_DATASET_DIR, get_gpt2_output_path
from molcrawl.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer as Tokenizer

# Medium-Sized GPT2 Model

n_layer = 24
n_head = 16
n_embd = 1024

dataset_dir = UNIPROT_DATASET_DIR

tensorboard_dir = get_gpt2_output_path("protein_sequence", "medium")
out_dir = get_gpt2_output_path("protein_sequence", "medium")

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 1 GPU = 61,440
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8  # match effective batch size to other domains

# this makes total number of tokens be 300B
max_iters = 50000
lr_decay_iters = 50000

# eval stuff
eval_interval = 1000
eval_iters = 200
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
learning_rate = 6e-5
min_lr = learning_rate / 10

# regularisation
weight_decay = 1e-1
dropout = 0.1

# dataset
dataset = "protein_sequence"

dataset_params = {
    "dataset_dir": dataset_dir  # Adjust the path as necessary for your generated dataset.
}
