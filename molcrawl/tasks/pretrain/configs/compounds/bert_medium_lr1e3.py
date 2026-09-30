# compounds BERT medium — learning-rate grid, point 3 of 3 (lr 0.001)
# launch: torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py <this config>
#
# The grid is shifted down with size rather than repeated. The learning rate a run
# collapses at falls as the model grows: across molnl, protein and rna the highest rate
# that still learned dropped by 1.73x to 3.16x per size step (measured from the
# 2026-09-17 grid, 30 runs). Holding one set of three rates across four sizes would put
# every arm of the largest size on the collapsing side.
#
# Steps of about 1.8x, with two points shared with each neighbouring size, so a wrong
# guess still leaves one grid straddling the optimum and the shared rates can be
# compared size against size:
#
#   small   5e-4    1e-3    2e-3     (run; 2e-3 collapsed)
#   medium  2.8e-4  5e-4    1e-3
#   large   1.5e-4  2.8e-4  5e-4
#   xl      8e-5    1.5e-4  2.8e-4
#
# Everything except learning_rate, model_size and the micro-batch split is identical to
# the small grid, and seed is 9 across all of them, so size and rate are the only
# things that move.

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 15000
warmup_steps = 1500  # 10% of max_steps, matching the small grid
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "medium"
# Per-arm directory under MODEL_OUTPUT_ROOT. Unset, this resolves under
# LEARNING_SOURCE_DIR -- the tree compounds' 18G of input and 1.4T of output already
# share -- and main.py's output guard stops the run before the first step.
model_path = get_bert_output_path("compounds", model_size) + "-lr1e3"
max_length = 1024  # packed blocks; sets BertConfig.max_position_embeddings
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# The compounds sets were packed in source order before the 2026-08-21 rebuild, so the
# split's leading rows are shorter and easier than the split as a whole. Draw the eval
# subset at random instead.
eval_subset_random = True
learning_rate = 0.001
weight_decay = 0.01
log_interval = 100  # = eval_steps -> 150 eval points over the run
save_steps = 1000  # multiple of eval_steps, so every checkpoint carries an eval

# Keep the checkpoint the reported number came from. At 150 eval points and 15 save
# points the minimum lands off the save grid nine times in ten, and the arm would be
# ranked on a number whose weights no longer exist.
save_on_improve = True

# Confine attention to one document inside a packed block. Run 53767 passed this at
# launch; without it a masked token attends across document boundaries.
document_masking = True

# 64 x 10 x 4 GPUs = 2,560 sequences. The split comes from the 2026-09-15
# micro-batch trial, which found 64 the largest per-device batch that fits at this
# size. That trial ran in fp32, before bf16 went into the BERT configs; bf16 halves the
# activations, so a larger split may well fit now. This is the shape attested to fit,
# not the one attested to be fastest -- the small trial found 128 x 5 some 36 % faster
# than the largest that fit.
batch_size = 64
gradient_accumulation_steps = 10

# The grid writes every key out rather than importing bert_small.py, so nothing added to
# the base reaches it -- these have to be stated here. Same values as every BERT base:
# four dataloader workers into pinned buffers, and bf16 (all-bert-order-2026-09-17 §1).
# A grid must not mix worker counts across its arms: the worker count changes the
# masked positions (§2).
dataloader_num_workers = 4
dataloader_pin_memory = True
bf16 = True
# 32 x 20 x 4 GPUs = 2,560. Declared so main.py stops a launch on any other GPU count
# instead of training at another batch (§5.3).
expected_global_batch = 2560
# Evaluation reads a fixed 10,000 rows (models/bert/main.py EVAL_SUBSET_ROWS), so an
# eval point costs the same work however it is batched -- but at a micro-batch far
# below the training one it takes far longer in wall time. genome measured the
# consequence: 8.08 s per eval against a 0.40 s training step, which is 16.3 % of the
# run at eval_interval=100 (commit 4972a37). Matching the training micro-batch is safe
# by construction: evaluation runs two no_grad forwards per batch (HF's own and the
# breakdown's in models/bert/_mlm_diagnostics.py), and two of those peak below one
# forward+backward at the same width, which training already does. Logits are the
# term that scales -- 64 x 1,024 x 616 vocab here.
per_device_eval_batch_size = 64

# Seed 9, the same value the small grid used, so the size grid differs in size and
# learning rate alone. The per-config sequential seeds (medium 8, large 7, xl 118)
# belong to the ladder configs, which this grid does not read.
seed = 9
