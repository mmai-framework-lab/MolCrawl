# protein BERT LR grid-fill — large, lr 2.5e-5 (below 1e-4, 2x steps down).
# protein-order-2026-09-25 §3: fill the grid where the best LR is at/near an edge.
# Base import + learning_rate override only; adam_beta2=0.999, warmup 3,353, max_steps
# 33,531 (9 epoch), 8x80 (eff 2,560), bf16 and the dataloader settings all come from the
# base bert_large.py. seed 42 for grid parity (base seed differs). Distinct out path per arm.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_large import *  # noqa: F401,F403

learning_rate = 2.5e-5
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 42                   # grid parity
model_path = get_bert_output_path("protein_sequence", "large") + "-lr2p5e5"
