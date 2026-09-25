"""mol_nl GPT-2 small on the 1,500-iteration grid schedule (2026-09-25 order §4).

The four runs this replaces raised the size and lowered the learning rate at the same
time, so nothing in them separates the effect of one from the other. The grid holds the
same three rates at every size, and holds the schedule too: one step is 2,560 sequences
of 1,024 tokens whatever the size, so an equal iteration count is an equal number of
tokens read.

1,500 iterations is 12.07 epochs of the 0.33 G-token corpus and 3.93 G tokens read. It is
set by the earliest turn in the 3,000-iteration runs: xl bottomed out at 1,900 and rose
after, so a comparison that runs past that point compares four models at different sides
of their own minimum.

lr_decay_iters matches max_iters. A rate that decays over a different length is a
different rate at every step but the first, whatever the peak says.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_small import *  # noqa: F401,F403

max_iters = 1500
lr_decay_iters = 1500        # decay over the schedule that is actually run
warmup_iters = 30            # 2% of max_iters, the convention the base config states
eval_interval = 50           # 30 evaluation points per run
seed = 42                    # one seed across the grid: the base configs vary it by size

# Every run reaches 1,500. Stopping a run at its own minimum would leave the sizes
# measured over different windows, which is what §4.2 compares them over.
early_stopping = False

# A GB200 supports it and train.py would pick it anyway; stated so the run does not
# depend on what the hardware happens to offer.
dtype = "bfloat16"
