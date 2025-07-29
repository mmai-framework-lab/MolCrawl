# config for training BERT on protein sequences using ESM tokenizer
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ python bert/main.py bert/configs/protein_sequence.py

from protein_sequence.dataset.tokenizer import EsmSequenceTokenizer


# Tokenizer instantiation
tokenizer = EsmSequenceTokenizer()

# Training configuration
max_steps = 600000
model_path = "runs_train_bert_protein_sequence"
max_length = 1024
dataset_dir = "fundamental_models_202407/uniprot/training_ready_hf_dataset"
learning_rate = 6e-6
weight_decay = 1e-1
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 1

gradient_accumulation_steps = 5 * 16
output_dir = "out-bert-protein-sequence"

# Model size configuration
# Choose between small, medium or large
model_size = "small"

# Protein sequence specific vocabulary size
# ESM tokenizer uses character-level tokenization for protein sequences
meta_vocab_size = len(tokenizer.get_vocab())
