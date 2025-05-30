# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

from compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer


tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 600000
model_path = "runs_train_bert_compounds"
max_length = 1024
dataset_dir = "/nasa/datasets/riken/projects/fundamental_models_202407/compounds/training_ready_hf_dataset"
learning_rate = 6e-6
weight_decay = 1e-1
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 1

gradient_accumulation_steps = 5 * 16

# Choose between small, medium or large
model_size = "small"
