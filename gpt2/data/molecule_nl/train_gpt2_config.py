# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer


tensorboard = True  # log training metrics to tensorboard
tensorboard_dir = "runs_train_gpt2_molecule_nl_small_6e-6wu200-6000-its"
out_dir = "out-molecule-nl-gpt2-small-6e-6wu200-6000-its"

tokenizer = Tokenizer()

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 8  # max size in koala
eval_batch_size = 16  # max size in koala
block_size = 1024
gradient_accumulation_steps = 5 * 16

# this makes total number of tokens be 300B
max_iters = 6000
lr_decay_iters = 6000
warmup_iters = 200  # how many steps to warm up for
learning_rate = 6e-6  # max learning rate
min_lr = learning_rate/10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 200

# weight decay
weight_decay = 1e-1

# dataset
dataset = "molecule_nl"

dataset_params = {
    "dataset_dir": "outputs/training_ready_hf_dataset"
}

# Special Tokens
start_instruction = 1
end_instruction = [518, 29914, 25580, 29962]
eos_token = 2  # eos
