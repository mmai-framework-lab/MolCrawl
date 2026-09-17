# protein BERT learning-rate grid — medium, point 2/3 (lr 3e-4). Grid arm; reproduces one run.
# The reference runs (case A/C esmopt, and the base bert_medium.py default LR) are NOT in the
# grid: they differ in warmup/seed/mask. Kept only as reference values.
#
# Everything except learning_rate is set to the fixed-optimizer + throughput recipe, stated
# here because the base bert_medium.py leaves adam_beta2 to models/bert/main.py
# defaults (0.95 = the COLLAPSED optimizer, root cause 2026-08-25) and uses the slow
# 8x80 micro-batch. These lines are required for the grid to measure the learning rate rather
# than collapse; they keep the effective global batch at 2,560. max_steps (33,531, 9 epochs)
# and warmup_steps (3,353) come from the base.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_medium import *  # noqa: F401,F403

learning_rate = 3e-4
adam_beta2 = 0.999          # REQUIRED: base default 0.95 collapses
batch_size = 160            # 160x4 throughput (base 8x80 is ~5.7x slower); eff-batch 2,560 unchanged
gradient_accumulation_steps = 4
seed = 42                   # grid parity (base medium seed differs)
document_masking = True
# Per-arm directory under MODEL_OUTPUT_ROOT, named from the config so the three learning
# rates of a size never share one. Unset, this resolves under LEARNING_SOURCE_DIR and
# main.py's output guard stops the run before the first step.
model_path = get_bert_output_path("protein_sequence", "medium") + "-lr3e4"
