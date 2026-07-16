# Llama-style decoder XL config for molecule_nat_lang.
# Created 2026-05-27 to round out the post-number_sample-fix retraining cohort.
# Mirrors gpt2_xl.py — only the output-path helper and HF-converter flag differ.

from molcrawl.core.paths import MOLECULE_NAT_LANG_DATASET_DIR, get_llama_output_path
from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from molcrawl.data.molecule_nat_lang.utils.vocab_guard import EXPECTED_VOCAB_SIZE_GPT2, check_vocab_size

# EX-Large-Sized Llama (n_layer=48, n_head=25, n_embd=1600 ≈ 1.5 B params)
n_layer = 48
n_head = 25
n_embd = 1600

dataset_dir = MOLECULE_NAT_LANG_DATASET_DIR

tensorboard = True
tensorboard_dir = get_llama_output_path("molecule_nat_lang", "xl")
out_dir = get_llama_output_path("molecule_nat_lang", "xl")

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size
check_vocab_size(meta_vocab_size, expected=EXPECTED_VOCAB_SIZE_GPT2)

# Effective batch matches gpt2_xl exactly: 2 * 1024 * 5 * 16 = 163,840 tokens/iter.
batch_size = 2
block_size = 1024
gradient_accumulation_steps = 5 * 16

# Training schedule
max_iters = 30000
lr_decay_iters = 30000
warmup_iters = 200
learning_rate = 1e-6
min_lr = learning_rate / 10

# Eval
eval_interval = 200
eval_iters = 200
log_interval = 200

# Resume + checkpoint policy
init_from = "resume"
save_hf_checkpoints = False  # Llama-style HF converter is invalid for this arch
always_save_checkpoint = True
save_checkpoint_steps = None
max_checkpoints = 5
keep_legacy_ckpt = True

early_stopping = True
early_stopping_patience = 10

weight_decay = 1e-1
dropout = 0.1  # follow gpt2 small/medium/large protein defaults — dropout=0.1 for FT-like behaviour

dataset = "molecule_nat_lang"
dataset_params = {"dataset_dir": dataset_dir}

print(f"Using vocab_size: {meta_vocab_size}")

bos_token_id = tokenizer.tokenizer.eos_token_id
eos_token_id = tokenizer.tokenizer.eos_token_id
pad_token_id = tokenizer.tokenizer.eos_token_id
