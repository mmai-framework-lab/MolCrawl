# compounds BERT medium — packed 1024 ladder
# launch: torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py <this config>
#
# 2026-09-15: the six fields below were what run 53767 (bert small, lr 1e-3) was
# actually launched with. They were passed at startup and never written back, so this
# file kept saying 1,558 steps at 8 x 80 with no document masking while the run that
# produced the reported numbers used none of that. A config that does not describe the
# run it produced cannot be used to reproduce it, and the next arm inherits the drift.

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

# v4 packed data (2026-08-05): train = 398,917 blocks x 1024, no padding.
# HF Trainer is per-device, so global batch = batch_size * grad_accum * n_GPUs
# = 8 * 80 * 4 = 2,560 sequences (assumes the 4-GPU launch the whole ladder uses).
# 15,000 steps at 2,560 = 96.3 epochs over the train split.
max_steps = 15000
warmup_steps = 1500  # 10% of max_steps, as run 53767 was launched, not the 2% convention
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "medium"
# Resolves under MODEL_OUTPUT_ROOT when that is set, else under LEARNING_SOURCE_DIR.
# main.py refuses a model_path that lands inside an input tree, so a run that leaves
# MODEL_OUTPUT_ROOT unset stops before the first step instead of writing 1.4T of
# checkpoints beside the corpus, which is how compounds got there.
model_path = get_bert_output_path("compounds", model_size)
max_length = 1024  # packed blocks; sets BertConfig.max_position_embeddings
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# The compounds sets were packed in source-parquet order, so the split's leading rows
# are shorter and easier than the split as a whole. Draw the eval subset at random
# instead. Off by default in main.py because protein / RNA / genome shuffle in prep and
# gain nothing from it.
eval_subset_random = True
# Unchanged from what this file has always carried. It is not a measured value at this
# size -- the 2026-09-15 learning-rate grid is what settles it -- and it is left alone
# here so this commit moves only the six fields run 53767 overrode.
learning_rate = 0.0001
weight_decay = 0.01
log_interval = 100  # = eval_steps -> 150 eval points over the run
save_steps = 1000  # multiple of eval_steps, so every checkpoint carries an eval

# Keep the checkpoint the reported number came from. Evaluation is 10x finer than
# saving, so the minimum lands off the save grid nine times in ten.
save_on_improve = True

# Confine attention to one document inside a packed block. Run 53767 passed this at
# launch; without it a masked token attends across document boundaries.
document_masking = True

# 8 x 80 x 4 GPUs = 2,560 sequences. This is the shipped shape, kept because it is
# the one attested to fit; it is not the one attested to be fastest. The 2026-09-15
# order's micro-batch trial measures the largest per-device batch that fits at this
# size and replaces these two numbers with it. Global batch stays 2,560 either way.
batch_size = 8
gradient_accumulation_steps = 80
per_device_eval_batch_size = 8

# Training seed (sequentially assigned across the tracked pretrain configs on
# 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 8
