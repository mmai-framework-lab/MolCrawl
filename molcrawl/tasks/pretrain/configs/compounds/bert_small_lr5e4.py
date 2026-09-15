# compounds BERT small — learning-rate grid point 0.0005 (2026-09-15 order rev2)
# launch: torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py <this config>
#
# Everything here except learning_rate matches run 53767 (the 1e-3 point of the same
# grid) as it was actually launched, not as bert_small.py stood committed: that run
# overrode six fields at startup and they were never written back. A grid whose arms
# differ in warmup, evaluation interval or micro-batch split does not measure the
# learning rate, so the values are fixed in the config and nothing is overridden at
# launch.

import os

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

# Outputs go outside the tree the inputs come from. get_bert_output_path() resolves
# under LEARNING_SOURCE_DIR, which is where compounds' 1.4T of run artifacts already
# sit beside its 18G of input; the 2026-09-14 order requires new runs to land elsewhere.
_runs_root = os.environ.get("COMPOUNDS_BERT_RUNS_ROOT")
if not _runs_root:
    raise SystemExit(
        "set COMPOUNDS_BERT_RUNS_ROOT to a runs root outside the input tree "
        "(the learning_source_*_compounds_packed directory)"
    )
model_path = os.path.join(_runs_root, "bert-small-lr5e4")

max_steps = 15000
warmup_steps = 1500  # 10% of max_steps, matching run 53767 rather than the 2% convention
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "small"
max_length = 1024  # packed blocks; sets BertConfig.max_position_embeddings
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# The compounds sets were packed in source order before the 2026-08-21 rebuild, so the
# split's leading rows are shorter and easier than the split as a whole. Draw the eval
# subset at random instead.
eval_subset_random = True
learning_rate = 0.0005
weight_decay = 0.01
log_interval = 500  # = eval_steps -> 30 eval points, matching run 53767
save_steps = 2500  # multiple of eval_steps, matching run 53767

# Confine attention to one document inside a packed block. Run 53767 passed this at
# launch; without it a masked token attends across document boundaries.
document_masking = True

# 32 x 20 x 4 GPUs = 2,560 sequences, the same split run 53767 used. The committed
# 8 x 80 reaches the same global batch by a different micro-batch shape.
batch_size = 32
gradient_accumulation_steps = 20
per_device_eval_batch_size = 8

# Same seed as run 53767 so the grid differs in the learning rate alone.
seed = 9
