# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from config.paths import CELLXGENE_DATASET_DIR

# Build the tokenizer using the WordLevel model
from rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

original_tokenizer = TranscriptomeTokenizer()

pre_tokenizer = Tokenizer(WordLevel(vocab=original_tokenizer.gene_token_dict, unk_token="[UNK]"))

# Wrap into Hugging Face tokenizer
tmp_tokenizer = PreTrainedTokenizerFast(tokenizer_object=pre_tokenizer, unk_token="[UNK]", pad_token="[PAD]")
tmp_tokenizer.unk_token = "[UNK]"
tmp_tokenizer.sep_token = "[SEP]"
tmp_tokenizer.pad_token = "<pad>"
tmp_tokenizer.cls_token = "[CLS]"
tmp_tokenizer.mask_token = "<mask>"

tmp_tokenizer.save_pretrained("custom_tokenizer")

tokenizer = AutoTokenizer.from_pretrained("custom_tokenizer")


max_steps = 600000
model_path = "runs_train_bert_rna"
max_length = 1024
dataset_dir = CELLXGENE_DATASET_DIR
learning_rate = 6e-6
weight_decay = 1e-1
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 1
gradient_accumulation_steps = 5 * 16


# Special Tokens
eos_token = 0  # eos


# Choose between small, medium or large
model_size = "small"
output_dir = "out-bert-rna"
