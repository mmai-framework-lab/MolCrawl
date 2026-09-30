# protein BERT LR grid-fill — large, lr 3.5e-5 (log-midpoint of the 2.5e-5..5e-5 boundary).
# protein-grid-followup-order-2026-09-30 §3: the collapse boundary for large sits between
# 2.5e-5 (survived, 2.6334) and 5e-5 (collapsed), and the best LR is just below the boundary,
# so probe the untested middle. sqrt(2.5e-5 * 5e-5) = 3.54e-5.
# Base import + learning_rate override only; adam_beta2=0.999, warmup 3,353, max_steps
# 33,531 (9 epoch), 8x80 (eff 2,560), bf16 and the dataloader settings all come from the
# base bert_large.py. seed 42 for grid parity (base seed differs). Distinct out path per arm.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_large import *  # noqa: F401,F403

learning_rate = 3.5e-5
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 42                   # grid parity
model_path = get_bert_output_path("protein_sequence", "large") + "-lr3p5e5"
