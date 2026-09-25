"""mol_nl GPT-2 large at learning rate 1.5e-4, on the 1,500-iteration grid.

large's best of the three points was 3e-4, the bottom of the grid, with nothing below it.
1.5e-4 is one more step of the grid's own factor of two, downward.

Everything but the learning rate comes from gpt2_large_1500.py, so this point is
comparable to the three already run. In particular deterministic_val_eval stays off:
it landed in train.py after those three ran, and turning it on here would measure this
point differently from the points it exists to be compared against.

min_lr follows the peak at a tenth of it, as every config in this modality has it.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_large_1500 import *  # noqa: F401,F403

learning_rate = 1.5e-4
min_lr = 1.5e-4 / 10
