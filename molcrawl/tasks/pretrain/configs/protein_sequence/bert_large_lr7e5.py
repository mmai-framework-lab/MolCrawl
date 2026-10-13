# protein BERT LR grid — large, lr 7e-5, seed 42 (bracket the collapse boundary from above).
# protein-order-2026-10-05 §4: large 5e-5 collapsed at seed 42 but survived at seed 1/17,
# so the boundary sits near 5e-5 and moves with the seed. Probe 7e-5 at seed 42/1/17 to
# place the upper edge. If all three collapse, the boundary is between 5e-5 and 7e-5 with a
# seed-dependent width; if any survives, large's best LR is higher still.
# Base import + learning_rate override only; adam_beta2=0.999, warmup 3,353, max_steps
# 33,531, 8x80 (eff 2,560), bf16 from bert_large. seed 42 for grid parity. Distinct out path.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_large import *  # noqa: F401,F403

learning_rate = 7e-5
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 42                   # grid parity
model_path = get_bert_output_path("protein_sequence", "large") + "-lr7e5"
