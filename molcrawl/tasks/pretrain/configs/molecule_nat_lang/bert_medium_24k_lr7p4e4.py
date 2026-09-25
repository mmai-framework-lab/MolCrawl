"""mol_nl BERT medium at 24,000 steps, learning rate 7.4e-4.

medium's best at 12,000 steps was 3e-4 and 1e-3 collapsed, so the best sits somewhere between them. The three new points divide that interval in four on a log scale, a step of 1.35.

Everything but the learning rate comes from bert_medium_24k.py, imported rather than
copied, so the points of one size cannot differ in anything else.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_medium_24k import *  # noqa: F401,F403

learning_rate = 7.4e-4
