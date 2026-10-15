"""mol_nl GPT-2 medium at 6e-4, with the validation set held fixed.

The grid's runs drew their validation batches again at every evaluation, which put 0.017
to 0.022 of noise on each point -- more than the differences between sizes. Re-scoring the
finished checkpoints on the whole split showed the seed spread was real rather than an
artefact of that, but the curves themselves still carry it, and a curve is what the loss
figures are drawn from.

deterministic_val_eval makes the validation pass walk a fixed, seeded prefix of the split
instead of resampling with replacement. Training's own sampling and the gradients are
untouched.

6e-4 did best for medium (0.5953, against 0.6315 at 3e-4 and 0.7753 at 1.2e-3).

Everything else comes from gpt2_medium_1500.py: 1,500 iterations, 2,560 sequences of
1,024 tokens per step, seed 42.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_medium_1500 import *  # noqa: F401,F403

learning_rate = 6e-4
min_lr = 6e-4 / 10

# §2 of the 2026-10-14 order: the validation set is fixed, not redrawn per evaluation.
deterministic_val_eval = True
