"""mol_nl BERT xl at 24,000 steps, learning rate 1.2e-4.

The middle: where the boundary is expected to be.

The boundary between the rate that holds and the rate that diverges has fallen with size:
1.14e-3 at small, 4.75e-4 at medium, 2.63e-4 at large (geometric middle of the bracketing
pair). Carrying that trend to xl puts it near 1.3e-4; these three points straddle it at
1.4x spacing, so the run says which side it is on whichever way the extrapolation errs.

Everything but the learning rate comes from bert_xl_24k.py.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_xl_24k import *  # noqa: F401,F403

learning_rate = 1.2e-4
