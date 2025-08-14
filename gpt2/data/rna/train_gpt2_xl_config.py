import sentencepiece as spm
from rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

# EX-Large-Sized GPT2 Model
n_layer = 48
n_head = 25
n_embd = 1600

tokenizer = TranscriptomeTokenizer()

tensorboard = True  # log training metrics to tensorboard

tensorboard_dir = "gpt2-output/rna-ex-large"
out_dir = "gpt2-output/rna-ex-large"

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

dataset_params = {"dataset_dir": "./fundamental_models_202407/cellxgene/training_ready_hf_dataset"}
