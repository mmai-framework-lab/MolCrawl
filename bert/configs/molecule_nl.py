# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer as Tokenizer
from config.paths import MOLECULE_NL_DATASET_DIR

tokenizer = Tokenizer()

max_steps = 600000
model_path = "runs_train_bert_molecule_nl"
max_length = 1024
dataset_dir = MOLECULE_NL_DATASET_DIR
learning_rate = 6e-6
weight_decay = 1e-1
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 1

gradient_accumulation_steps = 5 * 16


# Special Tokens
start_instruction = 1
end_instruction = [518, 29914, 25580, 29962]
eos_token = 2  # eos


# Choose between small, medium or large
model_size = "small"
output_dir = "out-bert-molecule-nl"
