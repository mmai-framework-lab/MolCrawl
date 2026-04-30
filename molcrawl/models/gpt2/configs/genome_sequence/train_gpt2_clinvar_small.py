# GPT-2 (small) fine-tuning config for ClinVar genome sequences
#
# Continues from the genome_sequence GPT-2 pretraining checkpoint using the
# ClinVar fine-tuning dataset (reference + variant sequences from human
# clinical variant annotations).
#
# Based on train_gpt2_small_config.py — key differences:
#   - dataset_dir / out_dir point to the ClinVar dataset and output
#   - pretrain_dir loads weights from genome_sequence pretraining
#   - max_iters reduced to 10000 (fine-tuning, not pretraining from scratch)
#   - learning_rate reduced to 1e-5

import sentencepiece as spm
from molcrawl.core.paths import (
    CLINVAR_DATASET_DIR,
    get_gpt2_output_path,
    get_refseq_tokenizer_path,
)

tokenizer_path = get_refseq_tokenizer_path()
dataset_dir = CLINVAR_DATASET_DIR

tensorboard_dir = get_gpt2_output_path("genome_sequence_clinvar", "small")
out_dir = get_gpt2_output_path("genome_sequence_clinvar", "small")
# Pretraining checkpoint to load weights from when out_dir has no checkpoint.
pretrain_dir = get_gpt2_output_path("genome_sequence", "small")

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
meta_vocab_size = tokenizer.vocab_size()

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

# Resume from genome_sequence pretraining checkpoint if available,
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

dataset = "genome_sequence_clinvar"

dataset_params = {
    "dataset_dir": dataset_dir,
}

# --- MolCrawl HF token IDs (added by patch_configs.py) ---
# SentencePiece: <unk>=0, <s>=1, </s>=2
bos_token_id = 1
eos_token_id = 2
pad_token_id = 0
