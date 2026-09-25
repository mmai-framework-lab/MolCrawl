"""mol_nl BERT large at 24,000 steps, learning rate 1.3e-4.

large's best at 12,000 steps was the bottom of the grid, 1e-4, and 3e-4 collapsed. The three new points divide 1e-4 to 3e-4 in four on a log scale, a step of 1.32.

Everything but the learning rate comes from bert_large_24k.py, imported rather than
copied, so the points of one size cannot differ in anything else.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_large_24k import *  # noqa: F401,F403

learning_rate = 1.3e-4
