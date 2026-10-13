# protein BERT xl LR grid — lr 3.4e-5 (upper point, 1.4x step above 2.4e-5).
# protein-order-2026-10-05 §5: the running xl points (1.2e-5/8e-6/5.5e-6) sit within 0.0004
# of each other at 48%, a sign all three are below the optimum. Add 1.7e-5/2.4e-5/3.4e-5
# (1.4x spacing) above them. seed 42 (added to the existing xl grid, which runs at seed 42).
# Base import + learning_rate override only; adam_beta2=0.999, warmup 3,353, max_steps
# 33,531, 8x80 (eff 2,560), bf16 from bert_xl. Distinct out path per arm.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_xl import *  # noqa: F401,F403

learning_rate = 3.4e-5
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 42                   # grid parity (existing xl grid is seed 42)
model_path = get_bert_output_path("protein_sequence", "xl") + "-lr3p4e5"
