"""mol_nl GPT-2 small at learning rate 2.4e-3, on the 1,500-iteration grid.

small's best of the three points was 1.2e-3, the top of the grid, with nothing above it
to say whether the best is there or further up. 2.4e-3 is one more step of the grid's
own factor of two.

Everything but the learning rate comes from gpt2_small_1500.py, so this point is
comparable to the three already run.

min_lr follows the peak at a tenth of it, as every config in this modality has it.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_small_1500 import *  # noqa: F401,F403

learning_rate = 2.4e-3
min_lr = 2.4e-3 / 10
