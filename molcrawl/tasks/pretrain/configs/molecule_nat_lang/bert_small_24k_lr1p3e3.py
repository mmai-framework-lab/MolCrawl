"""mol_nl BERT small at 24,000 steps, learning rate 1.3e-3.

small's ceiling is not yet bracketed. 1e-3 ran to the end of the 12,000-step grid and is
running again here; 2e-3 diverged at step 1,900, where warmup had the rate at
2e-3 x 1,900 / 2,400 = 1.58e-3, not at 2e-3. So the last rate known to hold is 1e-3 and
the first known to break is 1.58e-3, and nothing has been measured between them.

1.3e-3 is the log midpoint of that interval.

Everything but the learning rate comes from bert_small_24k.py, imported rather than
copied, so this point differs from the rest of the grid in one thing.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_small_24k import *  # noqa: F401,F403

learning_rate = 1.3e-3
