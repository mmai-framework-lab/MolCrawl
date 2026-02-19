# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py



from config.paths import MOLECULE_NL_DATASET_DIR, get_gpt2_output_path
from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer

# EX-Large-Sized GPT2 Model
n_layer = 48
n_head = 25
n_embd = 1600

dataset_dir = MOLECULE_NL_DATASET_DIR

tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = get_gpt2_output_path("molecule_nl", "xl")
out_dir = get_gpt2_output_path("molecule_nl", "xl")

tokenizer = Tokenizer()

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
init_from = "resume"  # 'scratch' or 'resume' - デフォルトでチェックポイントから再開

# checkpoint management
always_save_checkpoint = True  # 検証ロスに関係なく定期的に保存
save_checkpoint_steps = None  # Noneの場合はeval_intervalで保存
max_checkpoints = 5  # 最大5個のチェックポイントを保持

# weight decay
weight_decay = 1e-1

# dataset
dataset = "molecule_nl"

# Special Tokens
start_instruction = 1
end_instruction = [518, 29914, 25580, 29962]
eos_token = 2  # eos

dataset_params = {
    "dataset_dir": dataset_dir  # Adjust the path as necessary for your generated dataset.
}

# Vocabulary size for the model
try:
    if hasattr(tokenizer.tokenizer, "vocab_size"):
        meta_vocab_size = tokenizer.tokenizer.vocab_size
    else:
        meta_vocab_size = 32000  # CodeLlama default vocab size
except AttributeError:
    meta_vocab_size = 32000  # Fallback value

print(f"Using vocab_size: {meta_vocab_size}")
