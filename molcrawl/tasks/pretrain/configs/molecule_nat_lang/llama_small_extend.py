# Resume molecule_nat_lang × llama-small from iter 6000 with a constant LR.
# Mirror of gpt2_small_extend.py for the Llama-style decoder.

from molcrawl.core.paths import MOLECULE_NAT_LANG_DATASET_DIR, get_llama_output_path
from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from molcrawl.data.molecule_nat_lang.utils.vocab_guard import EXPECTED_VOCAB_SIZE_GPT2, check_vocab_size

tensorboard = True
tensorboard_dir = get_llama_output_path("molecule_nat_lang", "small")
out_dir = get_llama_output_path("molecule_nat_lang", "small")

dataset_dir = MOLECULE_NAT_LANG_DATASET_DIR

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size
check_vocab_size(meta_vocab_size, expected=EXPECTED_VOCAB_SIZE_GPT2)

batch_size = 8
block_size = 1024
gradient_accumulation_steps = 5 * 16

# Extension
init_from = "resume"
max_iters = 20000
lr_decay_iters = 20000
decay_lr = False
learning_rate = 3e-6
min_lr = learning_rate / 10
warmup_iters = 0

eval_interval = 200
eval_iters = 200
log_interval = 200

# Llama-style HF converter is invalid for this arch (see llama/train.py)
save_hf_checkpoints = False
save_checkpoint_steps = 1000
max_checkpoints = 10
keep_legacy_ckpt = True
always_save_checkpoint = False

early_stopping = True
early_stopping_patience = 10

weight_decay = 1e-1

dataset = "molecule_nat_lang"
dataset_params = {"dataset_dir": dataset_dir}

print(f"Using vocab_size: {meta_vocab_size}")

bos_token_id = tokenizer.tokenizer.eos_token_id
eos_token_id = tokenizer.tokenizer.eos_token_id
pad_token_id = tokenizer.tokenizer.eos_token_id
