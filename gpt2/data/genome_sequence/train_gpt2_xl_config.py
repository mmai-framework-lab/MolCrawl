# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# EX-Large-Sized GPT2 Model
n_layer = 48
n_head = 25
n_embd = 1600

import sentencepiece as spm

tokenizer_path = "learning_source_202508/refseq/spm_tokenizer.model"  # Adjust the path as necessary for your generated tokenizer.

dataset_dir = "learning_source_202508/refseq/training_ready_hf_dataset"  # Adjust the path as necessary for your generated dataset.

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = "gpt2-output/genome_sequence-xl"
out_dir = "gpt2-output/genome_sequence-xl"

tokenizer = spm.SentencePieceProcessor(
    model_file=tokenizer_path
)
meta_vocab_size = tokenizer.vocab_size()
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
warmup_iters = 200  # how many steps to warm up for
learning_rate = 6e-6  # max learning rate
min_lr = learning_rate / 10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# dataset
dataset = "genome_sequence"

dataset_params = {"dataset_dir": dataset_dir}
