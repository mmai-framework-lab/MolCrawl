# compounds BERT small — this size's best rate, rerun at seed 42
# launch: workflows/bert-grid.sbatch (SEGMENT=1, then continuations)
#
# The seed-42 rerun the directive of 2026-10-16 asks for. Identical to
# bert_small_lr1e3.py except for the seed and the output directory, so the pair
# measures the seed alone.
#
# Two fixes stand behind these runs, and the second only exists because the first
# was incomplete.
#
# save_on_improve watched on_evaluate, where eval_loss_mask has not arrived yet --
# MlmBreakdownMixin adds it after super().evaluate() returns -- so it fell back to
# eval_loss and the judged metric's minimum was not always saved. Measured on medium
# lr2.8e-4 (job 151323), whose minimum at step 14,700 has no checkpoint. It now
# watches on_log, where the metric does arrive.
#
# But the value arriving at on_log was each rank's own eval shard. The callback runs
# on every rank, so a step where one rank's local loss failed to improve sent the
# others into the save path alone, the collectives stopped lining up, and NCCL's
# watchdog took the job down. Six runs died that way before MlmBreakdownMixin was
# made to sum its totals over the ranks: 178504/178506/178508/178510 on 2026-10-16
# and 186947/186949 on 2026-10-19, small at step 2,300 and medium at step 2,700 on
# both attempts.
#
# seed 9 stays. Its runs are not re-run and not overwritten: the output directory
# here carries -seed42.

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 15000
warmup_steps = 1500  # 10% of max_steps, matching run 53767 rather than the 2% convention
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "small"
# Per-arm directory under MODEL_OUTPUT_ROOT. Unset, this resolves under
# LEARNING_SOURCE_DIR -- the tree compounds' 18G of input and 1.4T of output already
# share -- and main.py's output guard stops the run before the first step.
model_path = get_bert_output_path("compounds", model_size) + "-lr1e3-seed42"
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

# 32 x 20 x 4 GPUs = 2,560 sequences, the split run 53767 was launched with.
batch_size = 32
gradient_accumulation_steps = 20

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
# term that scales -- 32 x 1,024 x 616 vocab = 81 MB here.
per_device_eval_batch_size = 32

# seed 42 per the standing directive. data_seed is not written: neither path
# reads it and both refuse a config that declares one different from seed.
seed = 42

# HF's own default, stated here so the window is a choice rather than a leftover.
# It covers the largest checkpoint write -- 6.5 GB for large, about 18 GB for xl,
# written by rank 0 to Lustre while the other ranks wait at the next collective --
# and no save on record has come near it.
#
# It is deliberately not longer. The first two attempts at these runs were killed by
# a rank divergence in the save decision (see the header above), which no timeout
# cures: the second attempt ran at 7,200 s, logged Timeout(ms)=7200000 and died the
# same way, so the only thing the wider window bought was two hours of four idle
# GPUs instead of thirty minutes.
ddp_timeout = 1800
