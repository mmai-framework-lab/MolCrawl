# Resume protein_sequence × llama-small from iter 33000 with a constant LR.
# Mirror of gpt2_small_extend.py for the Llama-style decoder.

from molcrawl.core.paths import UNIPROT_DATASET_DIR, get_llama_output_path
from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer as Tokenizer

dataset_dir = UNIPROT_DATASET_DIR

tensorboard_dir = get_llama_output_path("protein_sequence", "small")
out_dir = get_llama_output_path("protein_sequence", "small")

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# Extension
init_from = "resume"
max_iters = 150000
lr_decay_iters = 150000
decay_lr = False
learning_rate = 1e-4
min_lr = learning_rate / 10
warmup_iters = 0

eval_interval = 1000
eval_iters = 200
log_interval = 10

save_hf_checkpoints = False   # Llama-style HF converter is invalid for this arch (see llama/train.py)
save_checkpoint_steps = 5000
max_checkpoints = 10
keep_legacy_ckpt = True
always_save_checkpoint = False

early_stopping = True
early_stopping_patience = 10

weight_decay = 1e-1
dropout = 0.1

dataset = "protein_sequence"
dataset_params = {"dataset_dir": dataset_dir}

bos_token_id = 0
eos_token_id = 2
pad_token_id = 1
