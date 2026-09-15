# GPT-2 (xl) fine-tuning config for RNA cell type annotation
#
# Continues from the rna GPT-2 pretraining checkpoint using the
# Geneformer cell type annotation dataset (human single-cell transcriptomes
# pre-tokenized as rank-value gene encodings from ctheodoris/Genecorpus-30M).
#
# Based on train_gpt2_celltype_small.py — key differences:
#   - n_layer/n_head/n_embd match the xl pretraining config
#   - pretrain_dir loads weights from rna-xl (ex-large) pretraining

from molcrawl.core.paths import (
    RNA_CELLTYPE_DATASET_DIR,
    get_gpt2_output_path,
)
from molcrawl.data.rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

tokenizer = TranscriptomeTokenizer()
meta_vocab_size = len(tokenizer)

n_layer = 48
n_head = 25
n_embd = 1600

tensorboard_dir = get_gpt2_output_path("rna_celltype", "xl")
out_dir = get_gpt2_output_path("rna_celltype", "xl")
# Pretraining checkpoint to load weights from when out_dir has no checkpoint.
pretrain_dir = get_gpt2_output_path("rna", "xl")

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# batch_size x gradient_accumulation_steps, the number max_iters was derived
# from. nanoGPT divides grad_accum by the world size before this is checked
# (models/gpt2/train.py:311 then :339), so the effective batch does not move
# with the GPU count -- unlike the BERT side. Declared so a config whose two
# factors stop multiplying to it refuses to start: genome trained at 640
# against an intended 2,560 and protein's LR pilots at 480, both found by
# reading a log afterwards.
expected_global_batch = 480


# Fine-tuning: much shorter run than pretraining (60000 → 10000 iters)
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 100

eval_interval = 200
eval_iters = 50
log_interval = 50

# Resume from rna pretraining checkpoint if available,
# otherwise start from scratch.
init_from = "resume"

always_save_checkpoint = False
save_checkpoint_steps = 200
max_checkpoints = 5

early_stopping = True
early_stopping_patience = 5

# Fine-tuning hyper-parameters (lower LR than pretraining 6e-6)
learning_rate = 1e-5
min_lr = learning_rate / 10
weight_decay = 1e-1
dropout = 0.1

dataset = "rna_celltype"

dataset_params = {
    "dataset_dir": RNA_CELLTYPE_DATASET_DIR,
}

# --- MolCrawl HF token IDs (added by patch_configs.py) ---
# WordLevel gene tokenizer: <pad>=0 (used as EOS in training concatenation)
bos_token_id = 0
eos_token_id = 0
pad_token_id = 0

# Training seed (sequentially assigned across the 117 tracked pretrain configs
# on 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 110
