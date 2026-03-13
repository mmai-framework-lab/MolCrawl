# BERT (small) fine-tuning config for ChEMBL
#
# Continues from the compounds BERT pretraining checkpoint using the ChEMBL
# fine-tuning dataset (canonical SMILES from ChEMBL 36).
#
# Based on molcrawl/bert/configs/compounds.py — only the dataset path,
# model output directory, learning rate and max_steps differ.

from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.config.paths import CHEMBL_DATASET_DIR, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

model_size = "small"
# Fine-tuning checkpoint output — separate from pretraining output
model_path = get_bert_output_path("compounds_chembl", model_size)
# Pretraining checkpoint to initialise weights from when no fine-tune checkpoint exists.
pretrain_model_path = get_bert_output_path("compounds", model_size)

max_length = 256
dataset_dir = CHEMBL_DATASET_DIR

# Fine-tuning hyper-parameters (lower LR and fewer steps than pretraining)
learning_rate = 1e-5
max_steps = 60000  # ~10 % of the 600k pretraining steps
weight_decay = 1e-1

log_interval = 100
save_steps = 100

batch_size = 8
per_device_eval_batch_size = 1
gradient_accumulation_steps = 5 * 16
