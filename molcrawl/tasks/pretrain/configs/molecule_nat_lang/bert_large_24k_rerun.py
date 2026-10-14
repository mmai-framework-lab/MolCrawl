"""mol_nl BERT large rerun at 1.7e-4, seed 42, with the evaluation mask held fixed.

The 2026-10-14 order asks for every size to be run again at seed 42 with the validation
set fixed. For BERT that is what bert_large_24k_lr1p7e4.py already did -- the mask has been
fixed since 2026-09-25 and the grid's seed is 42 -- so this run is a repetition of it
under a name of its own, which is what keeps it out of the finished run's directory.

1.7e-4 is large's best: 0.0581 as the mean of three seeds, with 1.3e-4 at 0.0614 and
2.3e-4 at 0.0584 either side of it.

Nothing is changed from bert_large_24k_lr1p7e4.py.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_large_24k_lr1p7e4 import *  # noqa: F401,F403
