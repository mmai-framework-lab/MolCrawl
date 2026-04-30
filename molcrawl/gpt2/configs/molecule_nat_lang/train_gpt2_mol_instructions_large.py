# GPT-2 (large) fine-tuning config for Mol-Instructions
#
# Continues from the molecule_nat_lang GPT-2 large pretraining checkpoint
# using the Mol-Instructions fine-tuning dataset.
#
# Based on train_gpt2_mol_instructions_small.py — key differences:
#   - n_layer/n_head/n_embd match the large pretraining config
#   - pretrain_dir loads weights from molecule_nat_lang-large pretraining

from molcrawl.core.paths import (
    MOL_INSTRUCTIONS_DATASET_DIR,
    get_gpt2_output_path,
)
from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from molcrawl.molecule_nat_lang.utils.vocab_guard import EXPECTED_VOCAB_SIZE_GPT2, check_vocab_size

n_layer = 36
n_head = 20
n_embd = 1280

tensorboard = True
tensorboard_dir = get_gpt2_output_path("molecule_nat_lang_mol_instructions", "large")
out_dir = get_gpt2_output_path("molecule_nat_lang_mol_instructions", "large")
# Pretraining checkpoint to load weights from when out_dir has no checkpoint.
pretrain_dir = get_gpt2_output_path("molecule_nat_lang", "large")

dataset_dir = MOL_INSTRUCTIONS_DATASET_DIR

# vocab_size is read dynamically from the tokenizer so that switching
# tokenizers (e.g. GPT-2 via GPT2_TOKENIZER_DIR) is reflected automatically.
tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size

# Batch / accumulation settings — same as pretraining config
batch_size = 2
block_size = 1024
gradient_accumulation_steps = 5 * 16

# Fine-tuning schedule: fewer iterations and a lower learning rate than
# the pretraining run (6000 iters @ 6e-6) to avoid catastrophic forgetting.
max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 100
learning_rate = 1e-5  # slightly higher than pretraining lr; adjust as needed
min_lr = learning_rate / 10

# Evaluation
eval_interval = 200
eval_iters = 200
log_interval = 50

# Resume from the molecule_nat_lang pretraining checkpoint
init_from = "resume"

# Checkpoint management
always_save_checkpoint = True
save_checkpoint_steps = None
max_checkpoints = 5

# Regularisation
weight_decay = 1e-1

# Dataset identifier used by the data-loader
dataset = "molecule_nat_lang_mol_instructions"

# Special tokens — GPT-2 tokenizer equivalents
start_instruction = 58  # first token of "[INST]"
end_instruction = [13412, 38604, 60]  # "[/INST]"
eos_token = 50256  # <|endoftext|>

dataset_params = {
    "dataset_dir": dataset_dir,
}

check_vocab_size(meta_vocab_size, expected=EXPECTED_VOCAB_SIZE_GPT2)
print(f"Using vocab_size: {meta_vocab_size}")

# MolCrawl HF token IDs — derived from the active GPT-2 tokenizer so the
# checkpoint config.json written by train.py always matches what the
# tokenizer actually emits. GPT-2 has no dedicated BOS/PAD tokens; the
# standard convention is to reuse <|endoftext|> (50256) for all three.
bos_token_id = tokenizer.tokenizer.eos_token_id
eos_token_id = tokenizer.tokenizer.eos_token_id
pad_token_id = tokenizer.tokenizer.eos_token_id
