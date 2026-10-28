# compounds BERT medium — the seed-42 run continued from step 30,000 to 45,000
# launch: workflows/bert-grid.sbatch with EXTEND=1, SEGMENT=2 and PREV_JOB=<the
#         completed 30,000-step job 206653>, from a COPY of
#         compounds-medium-lr2p8e4-seed42-ext30k/checkpoint-30000 with
#         scheduler.pt removed. See "The copy" below; without it this config's
#         learning rate and warmup are both ignored.
#
# The 30,000-step extension did not flatten: the improvement from 15,000 was
# -0.0017 and the minimum sits at step 29900 of 30,000. The 2026-10-28 judgement
# document chose plan ア -- all four sizes to 45,000, so that the schedule is the
# same across the ladder and the comparison across sizes stays a comparison of sizes.
#
# The rate is constant because HF decays over max_steps with no separate decay
# length: under the default "linear" a larger max_steps does not add steps to the
# existing curve, it recomputes the whole curve and continues a different run.
#
# The rate is UNCHANGED from the 30,000-step extension: 2.8e-5, a tenth of this
# size's original peak 2.8e-4. The 2026-10-28 directive §3 says the continuation runs
# at the same tenth and is not changed part-way, so that the whole 15,000-to-45,000
# stretch is one condition rather than three.
#
# The 500-step ramp is kept. The previous extension ended at a constant 2.8e-5 rather
# than at 0.0, so there is no jump from nothing at this resume point as there was at
# 15,000 -- but the ramp costs 500 steps out of 15,000 and removing it would make this
# arm's schedule shape differ from the one the 30,000-step values were measured under.
#
# The copy. HF restores the scheduler from the checkpoint, and
# LambdaLR.load_state_dict does self.__dict__.update(state_dict), which brings back
# base_lrs and last_epoch. get_lr returns base_lr * lambda(last_epoch), so resuming
# with that file present would run at the rate the previous segment ended on with the
# warmup already behind it, and the two values below would do nothing.
# bert-grid.sbatch refuses while scheduler.pt is present. optimizer.pt is kept: the
# moments are what a continuation needs.
#
# A separate output directory, so the 30,000-step run is untouched. Its checkpoints
# hold the values reported on 2026-10-27 and carried in the ladder until this run
# completes -- medium's checkpoint-29900 at 0.0676. Checkpoint retention keeps
# the best ten and would prune them if this extension wrote into the same directory.

from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer as Tokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT, get_bert_output_path

tokenizer = Tokenizer("assets/molecules/vocab.txt", 256)

max_steps = 45000
warmup_steps = 500  # the ramp at the resume point, not a fraction of max_steps
early_stopping = False  # Pretraining: run the full schedule, no early stopping
model_size = "medium"
# Per-arm directory under MODEL_OUTPUT_ROOT. Unset, this resolves under
# LEARNING_SOURCE_DIR -- the tree compounds' 18G of input and 1.4T of output already
# share -- and main.py's output guard stops the run before the first step.
model_path = get_bert_output_path("compounds", model_size) + "-lr2p8e4-seed42-ext45k"
max_length = 1024  # packed blocks; sets BertConfig.max_position_embeddings
dataset_dir = COMPOUNDS_DATASET_DIR_BERT
# The compounds sets were packed in source order before the 2026-08-21 rebuild, so the
# split's leading rows are shorter and easier than the split as a whole. Draw the eval
# subset at random instead.
eval_subset_random = True
learning_rate = 0.000028
weight_decay = 0.01
log_interval = 100  # = eval_steps -> 450 eval points over the run
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

# Seed 9, the same value the small grid used, so the size grid differs in size and
# learning rate alone. The per-config sequential seeds (medium 8, large 7, xl 118)
# belong to the ladder configs, which this grid does not read.
seed = 42

# HF's own default, stated here so the window is a choice rather than a leftover. It
# covers the largest checkpoint write -- measured by the mtime span inside one
# checkpoint, small 995 MB in 1 s and xl 16,947 MB in 18 s -- and no save on record has
# come near it.
#
# It is deliberately not longer. The runs aborted at save boundaries on 2026-10-16 and
# 2026-10-19 were killed by a rank divergence in the save decision, not by this window:
# the second attempt ran at 7,200 s, logged Timeout(ms)=7200000 and died the same way.
# No timeout cures that, and a wider one only delays the report -- which it did on
# 2026-10-20, when jobs 192553 and 192557 took two hours rather than thirty minutes to
# surface the same divergence because they carried 7,200 s.
ddp_timeout = 1800

# Constant rate with a short ramp; see the header. Passed through by
# models/bert/main.py only when a config sets it.
lr_scheduler_type = "constant_with_warmup"

# A collapse here would otherwise spend the whole 15,000 added steps at the
# degenerate level, as large seed 42 did at 2.8e-4 (job 187295). 2.395 is the
# degenerate floor for this corpus; the extension starts near 0.07 and the rise
# expected over the first thousand steps is nowhere near it.
degenerate_loss_threshold = 2.395
