# protein GPT-2 re-run — small, lr 1.2e-3 (small's current best), seed 42, fixed val.
# protein-order-2026-10-14d §1: re-run GPT-2 all sizes at seed 42 with the validation set
# fixed (deterministic_val_eval), at each size's current-best LR. Train loss is recorded by
# the logger (logging_*.csv train_loss column) alongside val_loss (§2). Distinct out_dir.
from molcrawl.core.paths import get_gpt2_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.gpt2_small import *  # noqa: F401,F403

learning_rate = 1.2e-3
min_lr = 1.2e-4             # peak/10
seed = 42
out_dir = get_gpt2_output_path("protein_sequence", "small") + "-rerun"
deterministic_val_eval = True
