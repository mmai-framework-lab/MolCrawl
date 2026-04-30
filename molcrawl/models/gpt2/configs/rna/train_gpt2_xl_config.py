import os


from molcrawl.core.paths import CELLXGENE_DATASET_DIR, RNA_DATASET_DIR, get_gpt2_output_path
from molcrawl.data.rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

# EX-Large-Sized GPT2 Model
n_layer = 48
n_head = 25
n_embd = 1600

tokenizer = TranscriptomeTokenizer()
meta_vocab_size = len(tokenizer)  # TranscriptomeTokenizer uses __len__ instead of vocab_size

tensorboard = True  # log training metrics to tensorboard

tensorboard_dir = get_gpt2_output_path("rna", "xl")
out_dir = get_gpt2_output_path("rna", "xl")

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 60000
lr_decay_iters = 60000
warmup_iters = 200  # how many steps to warm up for
learning_rate = 6e-6  # max learning rate
min_lr = learning_rate / 10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

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
early_stopping_patience = 5

# weight decay
weight_decay = 1e-1

# dataset
dataset = "rna"

# RNA specific parameters
rna_data_dir = CELLXGENE_DATASET_DIR
rna_vocab_file = os.path.join(RNA_DATASET_DIR, "gene_vocab.json")

dataset_params = {"dataset_dir": CELLXGENE_DATASET_DIR}

# --- MolCrawl HF token IDs (added by patch_configs.py) ---
# WordLevel gene tokenizer: <pad>=0 (used as EOS in training concatenation)
bos_token_id = 0
eos_token_id = 0
pad_token_id = 0
