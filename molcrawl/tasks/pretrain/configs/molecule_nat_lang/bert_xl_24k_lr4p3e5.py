"""mol_nl BERT xl at 24,000 steps, learning rate 4.3e-5.

One step down.

The first bracket missed from below: 1.7e-4, 1.2e-4 and 8.5e-5 all diverged, the lowest
of them reaching 3.7699 at step 1,900 before turning. So xl's boundary is under 8.5e-5,
which is more than 3.1 times below large's 2.63e-4 -- further than the N^-0.84 fitted
over small to large predicted (1.26e-4). These four points step down by 1.4 from 6e-5,
which is a quarter of large's boundary rather than the third that extrapolation gave.

Everything but the learning rate comes from bert_xl_24k.py.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_xl_24k import *  # noqa: F401,F403

learning_rate = 4.3e-5
