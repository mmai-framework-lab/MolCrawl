# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.config.paths import COMPOUNDS_DATASET_DIR, get_gpt2_output_path

# Large-Sized GPT2 Model

n_layer = 36
n_head = 20
n_embd = 1280

dataset_dir = COMPOUNDS_DATASET_DIR

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = get_gpt2_output_path("compounds", "large")
out_dir = get_gpt2_output_path("compounds", "large")

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)
meta_vocab_size = tokenizer.vocab_size

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 2  # max size in koala

block_size = 1024
gradient_accumulation_steps = 5 * 16

# this makes total number of tokens be 300B
max_iters = 30000
lr_decay_iters = 30000
warmup_iters = 200  # how many steps to warm up for
learning_rate = 6e-7  # max learning rate
min_lr = learning_rate / 10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 200

# init from checkpoint
init_from = "resume"  # 'scratch' or 'resume' - resume from checkpoint by default

# checkpoint management
always_save_checkpoint = True  # Save regularly regardless of validation loss
save_checkpoint_steps = None  # If None, save with eval_interval
max_checkpoints = 5  # Keep up to 5 checkpoints

# early stopping
early_stopping = True
early_stopping_patience = 5

# weight decay
weight_decay = 1e-1

# dataset
dataset = "compounds"

# Special Tokens
start_instruction = 12
eos_token = 12  # eos

dataset_params = {"dataset_dir": dataset_dir}
