# compounds BERT large — packed 1024 ladder
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
model_size = "large"
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
# Phase 1-5c (2026-07-16): 5e-5 -> 3e-5. The 22913 (5e-5) attempt was auto-aborted by
# the early-plateau detector at eval 6 (val=1.79 > 1.5 threshold), then 22918 (3e-5)
# COMPLETED healthy with min val 0.1766 -- matching bert-small 0.176 / bert-medium
# 0.144. The boss's 2026-07-16 reply promotes 3e-5 to the unified default across every
# modality's BERT large because it is the empirically-attested convergent value at 340M
# scale.
#
# 2026-09-15: written as a literal. It used to read
# float(os.environ.get("SUBSET_BERT_LARGE_LR", "0.00003")), so the value a run actually
# trained with did not have to be the one in the file, and nothing recorded which it
# was. The value itself is unchanged; the 2026-09-15 grid is what replaces it.
learning_rate = 0.00003
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
# Where the input is fetched, not what is computed: the same rows in the same
# order, pulled by four worker processes into pinned buffers instead of by the
# training process itself. main.py defaults both off (main.py:655-656), and with
# them off the Arrow read, the MLM draw and the document masking all sit on the
# critical path of every step. Measured on rna small at 8 x 80, 4 GPUs: in fp32
# these two settings alone take a step from 12.476 to 11.044 s (1.13x); with bf16
# as well it falls to 3.812 s (3.27x from where it started). bf16 without them is
# 0.94x. Neither half does much on its own.
#
# The masked positions are not the same as a 0-worker run: the collate runs in
# the worker, whose RNG PyTorch seeds per worker. The rate and the objective are
# unchanged and each setting reproduces itself, but the draw differs, so a run
# started with workers is not a continuation of one started without.
#
# Workers and pinning approved 2026-09-16 (all-bert-throughput-verdict-2026-09-16b);
# bf16 approved 2026-09-17 (all-bert-order-2026-09-17 §1). bf16 is the one of the three
# that changes numerics: every GPT-2 result and genome's BERT (bert_small_subset.py)
# already ran in bf16, so this aligns the remaining BERT modalities with them rather
# than introducing a new precision. Results from before this line ran in fp32 and are
# tabulated with a precision column (§1.4).
dataloader_num_workers = 4
dataloader_pin_memory = True
bf16 = True

# 8 x 80 x 4 GPUs = 2,560, the batch max_steps was derived from. Without this line
# main.py's global-batch check does not run, and a launch on a different GPU count
# trains at a different batch without stopping (all-bert-order-2026-09-17 §5.3).
expected_global_batch = 2560
per_device_eval_batch_size = 8

# Training seed (sequentially assigned across the tracked pretrain configs on
# 2026-08-03; boss directive to fix per-config seeds for reproducibility).
# Consumed by the runner via configurator; do NOT change once a run has started.
seed = 7
