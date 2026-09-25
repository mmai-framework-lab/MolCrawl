"""mol_nl BERT small at 24,000 steps, learning rate 1e-3.

small's best at 12,000 steps was the top of the grid, 1e-3, with nothing above it. 2e-3 opens that end.

Everything but the learning rate comes from bert_small_24k.py, imported rather than
copied, so the points of one size cannot differ in anything else.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_small_24k import *  # noqa: F401,F403

learning_rate = 1e-3
