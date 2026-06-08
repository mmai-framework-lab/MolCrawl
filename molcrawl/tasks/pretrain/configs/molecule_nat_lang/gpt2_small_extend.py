# Resume molecule_nat_lang × gpt2-small from iter 6000 with a constant LR.
# Base config max_iters was 6000 (the corpus is small, ~3.3M rows). The
# completed run showed a 0.33 nat train-val gap at step 6000 (overfit
# onset), so this extension is partly diagnostic: if val continues to
# improve we get more data-amount benefit; if val degrades, early_stop
# will fire and we keep the existing best-val ckpt.pt unchanged.

from molcrawl.core.paths import MOLECULE_NAT_LANG_DATASET_DIR, get_gpt2_output_path
from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from molcrawl.data.molecule_nat_lang.utils.vocab_guard import EXPECTED_VOCAB_SIZE_GPT2, check_vocab_size

tensorboard = True
tensorboard_dir = get_gpt2_output_path("molecule_nat_lang", "small")
out_dir = get_gpt2_output_path("molecule_nat_lang", "small")

dataset_dir = MOLECULE_NAT_LANG_DATASET_DIR

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size
check_vocab_size(meta_vocab_size, expected=EXPECTED_VOCAB_SIZE_GPT2)

# Same global batch as base config — 8 * 1024 * 5 * 16 = 655,360 tokens/iter.
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 5 * 16

# Extension
init_from = "resume"
max_iters = 20000             # 6000 → 20000 (3.3x)
lr_decay_iters = 20000
decay_lr = False
learning_rate = 3e-6
min_lr = learning_rate / 10
warmup_iters = 0

eval_interval = 200
eval_iters = 200
log_interval = 200

# Best-val gating + legacy ckpt for eval-script compatibility
always_save_checkpoint = False
save_checkpoint_steps = 1000
max_checkpoints = 10
keep_legacy_ckpt = True

early_stopping = True
early_stopping_patience = 10  # 10 * 200 = 2000 iter patience

weight_decay = 1e-1

dataset = "molecule_nat_lang"
dataset_params = {"dataset_dir": dataset_dir}

print(f"Using vocab_size: {meta_vocab_size}")

bos_token_id = tokenizer.tokenizer.eos_token_id
eos_token_id = tokenizer.tokenizer.eos_token_id
pad_token_id = tokenizer.tokenizer.eos_token_id
