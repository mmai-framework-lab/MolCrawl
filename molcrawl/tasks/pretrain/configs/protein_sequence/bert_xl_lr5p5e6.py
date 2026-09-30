# protein BERT xl LR grid — lr 5.5e-6 (lower).
# protein-grid-followup-order-2026-09-30 §4: xl collapse boundary predicted near
# large's 2.5e-5 / 3 ~= 8e-6. Probe 1.2e-5 (upper) / 8e-6 (predicted) / 5.5e-6 (lower),
# 1.45x steps. Shape 8x80 confirmed to fit xl in bf16 (job 150716, ~21 s/it).
# Base import + learning_rate override only; adam_beta2=0.999, warmup 3,353, max_steps
# 33,531, 8x80 (eff 2,560), bf16 from bert_xl. seed 42 for grid parity. Distinct out path.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_xl import *  # noqa: F401,F403

learning_rate = 5.5e-6
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 42                   # grid parity
model_path = get_bert_output_path("protein_sequence", "xl") + "-lr5p5e6"
