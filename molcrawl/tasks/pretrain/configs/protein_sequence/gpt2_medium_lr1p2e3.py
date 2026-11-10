# protein GPT-2 — medium, lr 1.2e-3. protein-order-2026-09-25 §4: probe one grid step
# above the swept ceiling 6e-4 (grid step is 2x: 1.5e-4/3e-4/6e-4; 1.2e-3 is the next).
# Base import + learning_rate/min_lr override only. max_iters/lr_decay_iters 33,531,
# warmup 671 (2%), shape (eff 2,560), block 1,024, dtype bfloat16 all from gpt2_medium.py.
# seed 1001 = the 6e-4 comparison arm's seed. Distinct out_dir per arm.
from molcrawl.core.paths import get_gpt2_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.gpt2_medium import *  # noqa: F401,F403

learning_rate = 1.2e-3
min_lr = 1.2e-4            # peak/10
# Training seed 42 (igarashi's order of 2026-09-30, applied across the tree on
# 2026-11-10). The 1.2e-3 arms ran at 1001 to match the 6e-4 arm; that pairing is
# recorded in those runs' manifests.
seed = 42
out_dir = get_gpt2_output_path("protein_sequence", "medium") + "-lr1p2e3"

# §5.2: use the fixed (deterministic) validation eval (protein-order-2026-09-25 §4-7).
# Requires the train.py flag from feat/protein-gpt2-fixed-eval (5b02c9a) to be in main
# first; the configurator rejects an unknown key otherwise.
deterministic_val_eval = True
