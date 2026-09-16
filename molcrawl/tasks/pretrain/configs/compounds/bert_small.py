# compounds BERT small — packed 1024 ladder
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
# = 32 * 20 * 4 = 2,560 sequences (assumes the 4-GPU launch the whole ladder uses).
# 15,000 steps at 2,560 = 96.3 epochs over the train split.
max_steps = 15000
warmup_steps = 1500  # 10% of max_steps, as run 53767 was launched, not the 2% convention
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "small"
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
# Run 53767's learning rate, as that run was launched. bert_small.py said 1e-4 while
# the run that produced the reported 0.0705 used 1e-3. The 2026-09-15 grid measures
# this properly; until it reports, this is the value with a run behind it.
learning_rate = 0.001
weight_decay = 0.01
log_interval = 100  # = eval_steps -> 150 eval points over the run
save_steps = 1000  # multiple of eval_steps, so every checkpoint carries an eval

# Keep the checkpoint the reported number came from. Evaluation is 10x finer than
# saving, so the minimum lands off the save grid nine times in ten.
save_on_improve = True

# Confine attention to one document inside a packed block. Run 53767 passed this at
# launch; without it a masked token attends across document boundaries.
document_masking = True

# 32 x 20 x 4 GPUs = 2,560 sequences, the split run 53767 was launched with. The
# committed 8 x 80 reached the same global batch by a slower micro-batch shape.
batch_size = 32
gradient_accumulation_steps = 20
# Where the input is fetched, not what is computed: the same rows in the same
# order, pulled by four worker processes into pinned buffers instead of by the
# training process itself. main.py defaults both off (main.py:655-656), and with
# them off the Arrow read, the MLM draw and the document masking all sit on the
# critical path of every step. Measured on rna small at 8 x 80, that input-side
# work was the larger part of the step, and moving it off was worth more than
# three times the read alone.
#
# The masked positions are not the same as a 0-worker run: the collate runs in
# the worker, whose RNG PyTorch seeds per worker. The rate and the objective are
# unchanged and each setting reproduces itself, but the draw differs, so a run
# started with workers is not a continuation of one started without.
#
# Approved 2026-09-16 (all-bert-throughput-verdict-2026-09-16b). bf16 was not:
# on its own it measured 0.94x, and it changes numerics rather than placement.
dataloader_num_workers = 4
dataloader_pin_memory = True
# Evaluation reads a fixed 10,000 rows (models/bert/main.py EVAL_SUBSET_ROWS), so an
# eval point costs the same work however it is batched -- but at a micro-batch far
# below the training one it takes far longer in wall time. genome measured the
# consequence: 8.08 s per eval against a 0.40 s training step, which is 16.3 % of the
# run at eval_interval=100 (commit 4972a37). Matching the training micro-batch is safe
# by construction: evaluation runs two no_grad forwards per batch (HF's own and the
# breakdown's in models/bert/_mlm_diagnostics.py), and two of those peak below one
# forward+backward at the same width, which training already does. Logits are the
# term that scales -- 32 x 1,024 x 616 vocab = 81 MB here.
per_device_eval_batch_size = 32

# Training seed (sequentially assigned across the tracked pretrain configs on
# 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 9
