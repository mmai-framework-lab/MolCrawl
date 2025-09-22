import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import sentencepiece as spm
from rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer
from config.paths import CELLXGENE_DATASET_DIR, get_gpt2_output_path

# Large-Sized GPT2 Model

n_layer = 36
n_head = 20
n_embd = 1280

tokenizer = TranscriptomeTokenizer()

tensorboard = True  # log training metrics to tensorboard

tensorboard_dir = get_gpt2_output_path("rna", "large")
out_dir = get_gpt2_output_path("rna", "large")

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
dataset = "rna"

dataset_params = {"dataset_dir": CELLXGENE_DATASET_DIR}
