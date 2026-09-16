# protein BERT learning-rate grid — large, point 1/3 (lr 1e-4). Grid arm; reproduces one run.
# The reference runs (case A/C esmopt, and the base bert_large.py default LR) are NOT in the
# grid: they differ in warmup/seed/mask. Kept only as reference values.
#
# Everything except learning_rate is set to the fixed-optimizer + throughput recipe, stated
# here because the base bert_large.py leaves adam_beta2/warmup_steps to models/bert/main.py
# defaults (0.95 / 200 = the COLLAPSED optimizer, root cause 2026-08-25) and uses the slow
# 8x80 micro-batch. These lines are required for the grid to measure the learning rate rather
# than collapse; they keep the effective global batch at 2,560.
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_large import *  # noqa: F401,F403

learning_rate = 1e-4
adam_beta2 = 0.999          # REQUIRED: base default 0.95 collapses
warmup_steps = 1118         # 10% of max_steps (11,177 = 3 epochs at global batch 2,560)
batch_size = 160            # 160x4 throughput (base 8x80 is ~5.7x slower); eff-batch 2,560 unchanged
gradient_accumulation_steps = 4
seed = 42                   # grid parity (base large seed differs)
document_masking = True
model_path = "/data1/rkp00024/matsubara/protein-bert-grid/bert-large-lr1e4"  # outside LEARNING_SOURCE_DIR (output_guard); distinct per arm
