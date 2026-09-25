# protein BERT LR grid-fill — medium, lr 1.3e-4 (between 1e-4 and 3e-4, log-quartered).
# protein-order-2026-09-25 §3: fill the grid where the best LR is at/near an edge.
# Base import + learning_rate override only; adam_beta2=0.999, warmup 3,353, max_steps
# 33,531 (9 epoch), 8x80 (eff 2,560), bf16 and the dataloader settings all come from the
# base bert_medium.py. seed 42 for grid parity (base seed differs). Distinct out path per arm.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_medium import *  # noqa: F401,F403

learning_rate = 1.3e-4
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 42                   # grid parity
model_path = get_bert_output_path("protein_sequence", "medium") + "-lr1p3e4"
