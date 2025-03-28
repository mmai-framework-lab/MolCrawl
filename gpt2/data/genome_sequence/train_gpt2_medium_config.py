# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor(
    model_file="/nasa/datasets/riken/projects/fundamental_models_202407/refseq/spm_tokenizer.model"
)

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = "runs_train_gpt2_genome_sequence"
out_dir = "out-genome_sequence"

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

dataset_params = {"dataset_dir": "/nasa/datasets/riken/projects/fundamental_models_202407/refseq/training_ready_hf_dataset"}
