# protein BERT learning-rate grid — medium, point 1/3 (lr 1e-4). Grid arm; reproduces one run.
# The reference runs (case A/C esmopt, and the base bert_medium.py default LR) are NOT in the
# grid: they differ in warmup/seed/mask. Kept only as reference values.
#
# The shape is the base's 8 x 80 (all-bert-order-2026-09-17 §4). 160 x 4 was measured
# against it in bf16 on large and ran out of GPU memory in both repetitions (jobs 122023,
# 122024); 8 x 80 ran at 11.548 and 11.484 s/step. max_steps (33,531, 9 epochs),
# warmup_steps (3,353), adam_beta2 (0.999) and the dataloader settings come from the base;
# adam_beta2 is restated below as well, so this file alone shows the grid is not on the
# collapsed optimizer.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_medium import *  # noqa: F401,F403

learning_rate = 1e-4
adam_beta2 = 0.999          # REQUIRED: base default 0.95 collapses
seed = 42                   # grid parity (base medium seed differs)
document_masking = True
# Per-arm directory under MODEL_OUTPUT_ROOT, named from the config so the three learning
# rates of a size never share one. Unset, this resolves under LEARNING_SOURCE_DIR and
# main.py's output guard stops the run before the first step.
model_path = get_bert_output_path("protein_sequence", "medium") + "-lr1e4"
