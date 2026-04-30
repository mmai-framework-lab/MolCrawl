# GPT-2 (small) fine-tuning config for ChEMBL
#
# Continues from the compounds GPT-2 pretraining checkpoint
# (see molcrawl/gpt2/configs/compounds/train_gpt2_small_config.py)
# using the ChEMBL fine-tuning dataset.
#
# Recommended launch command:
#   torchrun --standalone --nproc_per_node=<N> molcrawl/gpt2/train.py \
#       gpt2/configs/compounds/train_gpt2_chembl_small.py

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import CHEMBL_DATASET_DIR, get_gpt2_output_path

tensorboard = True
tensorboard_dir = get_gpt2_output_path("compounds_chembl", "small")
out_dir = get_gpt2_output_path("compounds_chembl", "small")
# Pretraining checkpoint to load weights from when out_dir has no checkpoint.
pretrain_dir = get_gpt2_output_path("compounds", "small")

tokenizer_path = "assets/molecules/vocab.txt"
tokenizer = Tokenizer(tokenizer_path, 256)
meta_vocab_size = tokenizer.vocab_size
eos_token_id = tokenizer.eos_token_id  # 13 ([SEP])

dataset_dir = CHEMBL_DATASET_DIR

# Batch / block settings — same as pretraining
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 5 * 16

# Fine-tuning schedule: fewer iterations and a lower LR than pretraining
# (pretraining: max_iters=6000, lr=6e-6).
max_iters = 2000
lr_decay_iters = 2000
warmup_iters = 100
learning_rate = 1e-5
min_lr = learning_rate / 10

# Evaluation
eval_interval = 200
eval_iters = 200
log_interval = 50

# Resume from compounds pretraining checkpoint
init_from = "resume"

# Checkpoint management
always_save_checkpoint = True
save_checkpoint_steps = None
max_checkpoints = 5

# Regularisation
weight_decay = 1e-1

# Dataset identifier used by the data-loader
dataset = "compounds_chembl"

# Special Tokens (SMILES tokenizer: [CLS]=2, [SEP]=3)
start_instruction = 2
eos_token = 2

dataset_params = {
    "dataset_dir": dataset_dir,
}
