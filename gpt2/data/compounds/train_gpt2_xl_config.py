# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

from compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer


# EX-Large-Sized GPT2 Model
n_layer = 48
n_head = 25
n_embd = 1600

dataset_dir = "outputs/compounds/training_ready_hf_dataset"  # path to the dataset directory

tokenizer_path = "assets/molecules/vocab.txt"  # path to the tokenizer vocab file

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = "gpt2-output/compounds-xl"
out_dir = "gpt2-output/compounds-xl"

tokenizer = Tokenizer(tokenizer_path, 256)

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
min_lr = learning_rate/10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 200

# weight decay
weight_decay = 1e-1

# dataset
dataset = "compounds"

# Special Tokens
start_instruction = 12
eos_token = 12  # eos

dataset_params = {
    "dataset_dir": dataset_dir
}