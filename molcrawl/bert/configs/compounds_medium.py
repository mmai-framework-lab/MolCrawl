# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 60000
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "medium"  # Choose between small, medium or large
model_path = get_bert_output_path("compounds", model_size)
max_length = 1024
dataset_dir = COMPOUNDS_DATASET_DIR
learning_rate = 6e-6
weight_decay = 1e-1
log_interval = 100

batch_size = 8
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 16
