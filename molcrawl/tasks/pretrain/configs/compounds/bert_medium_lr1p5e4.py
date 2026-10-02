# compounds BERT medium — learning-rate grid, point added below the original three
# launch: torchrun --standalone --nproc_per_node=4 molcrawl/models/bert/main.py <this config>
#
# Added 2026-10-01, after the first grid put medium's only surviving arm at the bottom
# of its own range. 2.8e-4 was still learning at 0.0711 while 5e-4 and 1e-3 both collapsed, so
# whether a lower rate does better is unmeasured.
#
# The rate a run collapses at falls faster in compounds than the three modalities the
# grid was extrapolated from: small's highest surviving rate is 1e-3 and medium's is
# 2.8e-4, a factor of 3.57 per size step, where molnl was 3.16, protein 3.00 and rna
# 1.73. The original grids were placed on that 1.73-3.16 range and sit too high.
#
# One step of about 1.8 below the surviving point, so the optimum is bracketed from
# below rather than pinned against the edge of the range.
#
# Seed 9, the seed this grid runs at. A point filled into an existing grid takes that
# grid's seed: a different one would put seed variance inside a difference that is
# supposed to be about the rate.

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
model_path = get_bert_output_path("compounds", model_size) + "-lr1p5e4"
max_length = 1024  # packed blocks; sets BertConfig.max_position_embeddings
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# The compounds sets were packed in source order before the 2026-08-21 rebuild, so the
# split's leading rows are shorter and easier than the split as a whole. Draw the eval
# subset at random instead.
eval_subset_random = True
learning_rate = 0.00015
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

# 160 x 4 x 4 GPUs = 2,560 sequences. Measured under bf16 on 2026-09-30 (job 150892):
# 320 does not fit, 160 does, and 160 is also the fastest of the shapes that fit
# (7.21 s/step against 7.35 at 64 and 14.47 at the shipped 8 x 80). The earlier trial
# said 64 because it ran in fp32, before bf16 went into the BERT configs; bf16 halves
# the activations and 160 now fits.
batch_size = 160
gradient_accumulation_steps = 4

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
# term that scales -- 160 x 1,024 x 616 vocab here.
per_device_eval_batch_size = 160

# seed only. data_seed is not written: neither trainer reads it -- transformers 4.45.1
# stores it on TrainingArguments and never consults it, and the data order follows
# set_seed(seed) -- so a declaration is dead config that reads as a second knob.
# main.py refuses a config whose data_seed disagrees with seed, for the same reason.
seed = 9
