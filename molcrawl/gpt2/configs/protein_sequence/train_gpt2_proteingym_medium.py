# GPT-2 (medium) fine-tuning config for ProteinGym DMS sequences
#
# Continues from the protein_sequence GPT-2 medium pretraining checkpoint using
# the ProteinGym fine-tuning dataset (mutated + wild-type sequences from DMS assays).
#
# Based on train_gpt2_proteingym_small.py — key differences:
#   - n_layer/n_head/n_embd match the medium pretraining config
#   - pretrain_dir loads weights from protein_sequence-medium pretraining


from molcrawl.core.paths import PROTEINGYM_DATASET_DIR, get_gpt2_output_path
from molcrawl.protein_sequence.dataset.tokenizer import (
    EsmSequenceTokenizer as Tokenizer,
)

dataset_dir = PROTEINGYM_DATASET_DIR

n_layer = 24
n_head = 16
n_embd = 1024

tensorboard_dir = get_gpt2_output_path("protein_sequence_proteingym", "medium")
out_dir = get_gpt2_output_path("protein_sequence_proteingym", "medium")
# Pretraining checkpoint to load weights from when out_dir has no checkpoint.
pretrain_dir = get_gpt2_output_path("protein_sequence", "medium")

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size  # 33 (character-level amino acid vocab)

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# Fine-tuning: much shorter run than pretraining (600k → 10000 iters)
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 100  # must be < lr_decay_iters to avoid ZeroDivisionError in get_lr

eval_interval = 200
eval_iters = 50
log_interval = 50

# Resume from protein_sequence pretraining checkpoint if available,
# otherwise start from scratch.
init_from = "resume"

always_save_checkpoint = False
save_checkpoint_steps = 200
max_checkpoints = 5

early_stopping = True
early_stopping_patience = 5

# Fine-tuning hyper-parameters (lower LR than pretraining 6e-4)
learning_rate = 1e-5
weight_decay = 1e-1
dropout = 0.1

dataset = "protein_sequence_proteingym"

dataset_params = {
    "dataset_dir": dataset_dir,
}

# --- MolCrawl HF token IDs (added by patch_configs.py) ---
# EsmSequenceTokenizer: <cls>=0, <pad>=1, <eos>=2
bos_token_id = 0
eos_token_id = 2
pad_token_id = 1
