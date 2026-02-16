# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from config.paths import UNIPROT_DATASET_DIR, get_gpt2_output_path
from protein_sequence.dataset.tokenizer import EsmSequenceTokenizer as Tokenizer

dataset_dir = UNIPROT_DATASET_DIR

tensorboard_dir = get_gpt2_output_path("protein_sequence", "small")
out_dir = get_gpt2_output_path("protein_sequence", "small")

tokenizer = Tokenizer()
meta_vocab_size = tokenizer.vocab_size

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# init from checkpoint
init_from = "resume"  # 'scratch' or 'resume' - デフォルトでチェックポイントから再開

# checkpoint management - 定期保存で過学習前のモデルを確保
always_save_checkpoint = False  # best modelのみ保存（過学習対策）
save_checkpoint_steps = 5000  # 5000ステップごとに定期保存
max_checkpoints = 10  # 過学習前のcheckpointを保持するため多めに

# early stopping - 過学習を検知して自動停止
early_stopping = True
early_stopping_patience = 5  # 5回（5000ステップ）改善がなければ停止

# regularization - 過学習を抑制
weight_decay = 1e-1
dropout = 0.1  # Dropoutを有効化（デフォルトは0.0）

# dataset
dataset = "protein_sequence"

dataset_params = {
    "dataset_dir": dataset_dir  # Adjust the path as necessary for your generated dataset.
}
