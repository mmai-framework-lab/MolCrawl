# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


from molcrawl.config.paths import MOLECULE_NAT_LANG_DATASET_DIR, get_gpt2_output_path
from molcrawl.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from molcrawl.molecule_nat_lang.utils.vocab_guard import EXPECTED_VOCAB_SIZE_GPT2, check_vocab_size

# EX-Large-Sized GPT2 Model
n_layer = 48
n_head = 25
n_embd = 1600

dataset_dir = MOLECULE_NAT_LANG_DATASET_DIR

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = get_gpt2_output_path("molecule_nat_lang", "xl")
out_dir = get_gpt2_output_path("molecule_nat_lang", "xl")

tokenizer = Tokenizer()
# GPT-2 tokenizer (vocab_size=50257) — nanoGPT configs use the raw size.
# check_vocab_size() below fails fast if a different tokenizer is loaded.
meta_vocab_size = tokenizer.vocab_size
check_vocab_size(meta_vocab_size, expected=EXPECTED_VOCAB_SIZE_GPT2)

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 2  # max size in koala

block_size = 1024
gradient_accumulation_steps = 5 * 16

# training
max_iters = 30000
lr_decay_iters = 30000
warmup_iters = 200  # how many steps to warm up for
learning_rate = 1e-6  # max learning rate
min_lr = learning_rate / 10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 200

# init from checkpoint
init_from = "resume"  # 'scratch' or 'resume' - resume from checkpoint by default

# checkpoint management
always_save_checkpoint = True  # Save regularly regardless of validation loss
save_checkpoint_steps = None  # If None, save with eval_interval
max_checkpoints = 5  # Keep up to 5 checkpoints

# early stopping
early_stopping = True
early_stopping_patience = 10  # increased from 5 to allow larger models more time to converge

# weight decay
weight_decay = 1e-1

# dataset
dataset = "molecule_nat_lang"

# Special Tokens
start_instruction = 1
end_instruction = [518, 29914, 25580, 29962]
eos_token = 2  # eos

dataset_params = {
    "dataset_dir": dataset_dir  # Adjust the path as necessary for your generated dataset.
}

print(f"Using vocab_size: {meta_vocab_size}")

# --- MolCrawl HF token IDs (added by patch_configs.py) ---
# NOTE: these values predate the GPT-2 tokenizer migration and have not yet
# been realigned with GPT-2 semantics (eos=50256, no dedicated pad). Do not
# treat as authoritative without cross-checking the GPT-2 training loop.
bos_token_id = 0
eos_token_id = 2
pad_token_id = 0
