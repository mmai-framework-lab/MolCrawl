# Llama-style decoder XL config for genome_sequence.
# Created 2026-05-27 to round out the post-number_sample-fix retraining cohort.
# Mirrors gpt2_xl.py — only the output-path helper and HF-converter flag differ.

import sentencepiece as spm

from molcrawl.core.paths import (
    REFSEQ_DATASET_DIR,
    get_llama_output_path,
    get_refseq_tokenizer_path,
)

# EX-Large-Sized Llama (n_layer=48, n_head=25, n_embd=1600 ≈ 1.5 B params)
n_layer = 48
n_head = 25
n_embd = 1600

tokenizer_path = get_refseq_tokenizer_path()
dataset_dir = REFSEQ_DATASET_DIR

tensorboard = True
tensorboard_dir = get_llama_output_path("genome_sequence", "xl")
out_dir = get_llama_output_path("genome_sequence", "xl")

tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
meta_vocab_size = tokenizer.vocab_size()

# Effective batch matches gpt2_xl: 12 * 1024 * 5 * 8 = 491,520 tokens/iter.
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

max_iters = 50000
lr_decay_iters = 50000
warmup_iters = 200
learning_rate = 6e-6
min_lr = learning_rate / 10

eval_interval = 1000
eval_iters = 200
log_interval = 10

init_from = "resume"
save_hf_checkpoints = False  # Llama-style HF converter is invalid for this arch
always_save_checkpoint = True
save_checkpoint_steps = None
max_checkpoints = 5
keep_legacy_ckpt = True

early_stopping = True
early_stopping_patience = 5

weight_decay = 1e-1

dataset = "genome_sequence"
dataset_params = {"dataset_dir": dataset_dir}

# SentencePiece: <unk>=0, <s>=1, </s>=2
bos_token_id = 1
eos_token_id = 2
pad_token_id = 0
