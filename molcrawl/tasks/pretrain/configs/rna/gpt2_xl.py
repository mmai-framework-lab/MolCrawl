import os


from molcrawl.core.paths import CELLXGENE_DATASET_DIR, PROJECT_ROOT, RNA_DATASET_DIR, get_gpt2_output_path
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

# Effective global batch = 2560 sequences (spec; same value as protein/genome/
# compounds so the modalities stay compute-matched). For GPT-2/nanoGPT the
# effective batch = batch_size * gradient_accumulation_steps and is GPU-count-
# independent (train.py does grad_accum //= world_size). Micro-batch is lowered
# at this size to stay memory-safe: 4 * 640 = 2560.
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 640  # 4 * 640 = 2560 seq global batch

# 3 epochs of the train split at global batch 2560:
# floor(3 * 34,407,040 train blocks / 2560) = 40,320 iters (~105.7B tokens processed).
max_iters = 40320
lr_decay_iters = 40320
warmup_iters = 806  # ~2% of max_iters (compounds convention); < max_iters so LR reaches peak
learning_rate = 0.0002  # max learning rate
min_lr = 2e-05  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 1000
eval_iters = 200
# eval_sequences fixes the *number of validation sequences* per eval point instead of
# the number of batches, so every ladder size averages its val loss over the same
# 3,200 sequences. batch_size shrinks with model size, so the old shared
# eval_iters=200 gave the large models a 2-4x smaller val sample.
eval_sequences = 3200
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
weight_decay = 0.1

# dataset
dataset = "rna"

# RNA specific parameters
rna_data_dir = CELLXGENE_DATASET_DIR

# Packed uint16 memmap of the same splits (built by rna_pack_to_bin.py).
# Identical rows in identical order — index i is index i — so this changes
# storage only, not the data or the sampling. Fetching a micro-batch out of
# Arrow costs ~175 ms with four ranks reading concurrently, which left RNA
# GPT-2 at ~2 % MFU with 85 % of each iteration waiting on data (job 19223).
# Comment this out to fall back to the Arrow path.
rna_bin_dir = RNA_DATASET_DIR + "/training_ready_bin"
# Geneformer token space (25,426: <pad>=0, <mask>=1, then Ensembl gene IDs) — the
# vocabulary training_ready is actually tokenized in (verified max token id 25,401).
# NOT rna/gene_vocab.json, which is the CellxGene census gene-symbol list (60,664,
# written for stats only); train.py derives the model vocab_size from this file via
# RNADataset, so pointing at the census list built GPT-2 with 58% unreachable rows.
rna_vocab_file = os.path.join(
    PROJECT_ROOT, "molcrawl", "data", "rna", "dataset", "geneformer", "token_dictionary.pkl"
)

dataset_params = {"dataset_dir": CELLXGENE_DATASET_DIR}

# --- MolCrawl HF token IDs (added by patch_configs.py) ---
# WordLevel gene tokenizer: <pad>=0 (used as EOS in training concatenation)
bos_token_id = 0
eos_token_id = 0
pad_token_id = 0

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 114
