# compounds BERT xlarge — the seed-42 run continued from step 15,000 to 30,000
# launch: workflows/bert-grid.sbatch with EXTEND=1, SEGMENT=2 and PREV_JOB=<the
#         completed 15,000-step job>, from a COPY of checkpoint-15000 with
#         scheduler.pt removed. Without that copy this config's learning rate and
#         warmup are both ignored -- see "The copy" below.
#
# This extension exists to see whether the loss flattens past the original schedule.
# It is not a ladder value unless the result says it should be.
#
# The rate is constant because HF decays over max_steps with no separate decay
# length: under the default "linear" a larger max_steps does not add steps to the
# existing curve, it recomputes the whole curve and continues a different run.
#
# constant_with_warmup, 500 steps, to 4.5e-6 -- a tenth of this size's peak 4.5e-5.
# The tenth and the 500 steps follow the shape matsubara set for small and medium on
# 2026-10-20; the directive of 2026-10-23 names the values for those two only, and
# matsubara applied the same rule to this size on 2026-10-23. The original schedule
# ends at exactly 0.0, so without a ramp the rate jumps from nothing to the new value
# at the resume point.
#
# The copy. HF restores the scheduler from the checkpoint, and
# LambdaLR.load_state_dict does self.__dict__.update(state_dict), which brings back
# base_lrs and last_epoch. get_lr returns base_lr * lambda(last_epoch). Read off a
# real checkpoint-15000/scheduler.pt: base_lrs [0.001, 0.001], last_epoch 15000. So
# resuming with that file present gives lambda(15000) = 1.0 past the 500-step warmup,
# multiplied by the ORIGINAL peak -- the rate below would do nothing. bert-grid.sbatch
# refuses while scheduler.pt is present. optimizer.pt is kept: the moments are what a
# continuation needs.
#
# A separate output directory, so the 15,000-step run is untouched. Checkpoint
# retention keeps the best ten and would otherwise prune the checkpoint the reported
# ladder value refers to.

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 30000
warmup_steps = 500  # the ramp at the resume point, not a fraction of max_steps
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "xl"
# Per-arm directory under MODEL_OUTPUT_ROOT. Unset, this resolves under
# LEARNING_SOURCE_DIR -- the tree compounds' 18G of input and 1.4T of output already
# share -- and main.py's output guard stops the run before the first step.
model_path = get_bert_output_path("compounds", model_size) + "-lr4p5e5-seed42-ext30k"
max_length = 1024  # packed blocks; sets BertConfig.max_position_embeddings
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# The compounds sets were packed in source order before the 2026-08-21 rebuild, so the
# split's leading rows are shorter and easier than the split as a whole. Draw the eval
# subset at random instead.
eval_subset_random = True
learning_rate = 4.5e-06
weight_decay = 0.01
log_interval = 100  # = eval_steps -> 150 eval points over the run
save_steps = 1000  # multiple of eval_steps, so every checkpoint carries an eval

# Keep the checkpoint the reported number came from. At 150 eval points and 15 save
# points the minimum lands off the save grid nine times in ten, and the arm would be
# ranked on a number whose weights no longer exist.
save_on_improve = True

# Stop a run that has fallen into the degenerate solution instead of letting it spend
# the rest of the schedule there. 2.395 is the degenerate floor measured for this
# corpus -- unigram 2.635, unigram plus the copyable 10 per cent 2.395, left-neighbour
# table 2.066 -- so a [MASK] loss that stays above it has not learned context.
#
# Measured on large seed 42 (job 187295): the loss reached 0.2389 at step 1,400, then
# step 1,500 0.5287, step 1,600 1.6868 with grad_norm 6.422, and from step 1,700 it sat
# between 2.47 and 2.87 through step 7,000. The collapse happened as warmup ended and
# the rate reached its peak. With the threshold unset the detector only warned, so the
# run went on spending four GPUs on the degenerate solution; at the default grace of
# 5,000 steps it would have stopped around there instead of at 15,000.
#
# The grad_norm warning is not a substitute. It fired on medium (8.029) and xl (11.673),
# which are both healthy, and large's 4.587 was the smallest of the three. What
# separated them was the grad_norm as warmup ended: small 0.190, medium 0.505, xl 3.090,
# large 6.422.
degenerate_loss_threshold = 2.395

# Confine attention to one document inside a packed block. Run 53767 passed this at
# launch; without it a masked token attends across document boundaries.
document_masking = True

# 32 x 20 x 4 GPUs = 2,560 sequences. Measured under bf16 on 2026-09-30:
# 64 does not fit, 32 does, and 32 is also the fastest of the shapes that fit
# (23.70 s/step against 30.31 at 16 and 43.72 at the shipped 8 x 80).
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
# term that scales -- 32 x 1,024 x 616 vocab here.
per_device_eval_batch_size = 32

# seed only. data_seed is not written: neither trainer reads it -- transformers 4.45.1
# stores it on TrainingArguments and never consults it, and the data order follows
# set_seed(seed) -- so a declaration is dead config that reads as a second knob.
# main.py refuses a config whose data_seed disagrees with seed, for the same reason.
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

# Constant rate with a short ramp; see the header. Passed through by
# models/bert/main.py only when a config sets it.
lr_scheduler_type = "constant_with_warmup"
