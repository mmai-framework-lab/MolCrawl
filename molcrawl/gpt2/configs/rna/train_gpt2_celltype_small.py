# GPT-2 (small) fine-tuning config for RNA cell type annotation
#
# Continues from the rna GPT-2 pretraining checkpoint using the
# Geneformer cell type annotation dataset (human single-cell transcriptomes
# pre-tokenized as rank-value gene encodings from ctheodoris/Genecorpus-30M).
#
# Based on train_gpt2_small_config.py — key differences:
#   - dataset_dir / out_dir point to the celltype dataset and output
#   - pretrain_dir loads weights from rna pretraining
#   - max_iters reduced to 10000 (fine-tuning, not pretraining from scratch)
#   - learning_rate reduced to 1e-5

from molcrawl.core.paths import (
    RNA_CELLTYPE_DATASET_DIR,
    get_gpt2_output_path,
)
from molcrawl.rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

tokenizer = TranscriptomeTokenizer()
meta_vocab_size = len(tokenizer)

tensorboard_dir = get_gpt2_output_path("rna_celltype", "small")
out_dir = get_gpt2_output_path("rna_celltype", "small")
# Pretraining checkpoint to load weights from when out_dir has no checkpoint.
pretrain_dir = get_gpt2_output_path("rna", "small")

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# Fine-tuning: much shorter run than pretraining (600k → 10000 iters)
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
