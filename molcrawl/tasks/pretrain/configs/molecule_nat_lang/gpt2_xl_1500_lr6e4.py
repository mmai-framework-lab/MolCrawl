"""mol_nl GPT-2 xl at learning rate 6e-4, on the 1,500-iteration grid.

Everything but the learning rate comes from gpt2_xl_1500.py, imported rather than
copied, so the three points of a size differ in one thing.

min_lr follows the peak at a tenth of it, as every config in this modality has it.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_xl_1500 import *  # noqa: F401,F403

learning_rate = 6e-4
min_lr = 6e-4 / 10
