# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


import sentencepiece as spm

from molcrawl.config.paths import (
    REFSEQ_DATASET_DIR,
    get_gpt2_output_path,
    get_refseq_tokenizer_path,
)


n_layer = 24
n_head = 16
n_embd = 1024

tokenizer_path = get_refseq_tokenizer_path()

dataset_dir = REFSEQ_DATASET_DIR

out_dir = "out-genome-sequence"  # output directory for model checkpoints

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = get_gpt2_output_path("genome_sequence", "medium")
out_dir = get_gpt2_output_path("genome_sequence", "medium")

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
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

# init from checkpoint
init_from = "resume"  # 'scratch' or 'resume' - デフォルトでチェックポイントから再開

# checkpoint management
always_save_checkpoint = True  # 検証ロスに関係なく定期的に保存
save_checkpoint_steps = None  # Noneの場合はeval_intervalで保存
max_checkpoints = 5  # 最大5個のチェックポイントを保持

# weight decay
weight_decay = 1e-1

# dataset
dataset = "genome_sequence"

dataset_params = {"dataset_dir": dataset_dir}
