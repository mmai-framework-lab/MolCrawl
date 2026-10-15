# protein BERT re-run — large, lr 3.5e-5 (large's current best at seed 42), seed 42.
# protein-order-2026-10-14d §1: re-run BERT large at seed 42 at its current-best LR. 5e-5
# collapses at seed 42, so 3.5e-5 is the highest non-collapsing rate (2.5851). BERT records
# both train loss and eval_loss_mask in trainer_state (§2). Distinct out path from the
# original -lr3p5e5 run so the segment guard does not refuse a fresh start.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_large import *  # noqa: F401,F403

learning_rate = 3.5e-5
adam_beta2 = 0.999          # restated (base default 0.95 collapses)
seed = 42                   # grid parity / re-run of the seed-42 arm
model_path = get_bert_output_path("protein_sequence", "large") + "-lr3p5e5-rerun"
