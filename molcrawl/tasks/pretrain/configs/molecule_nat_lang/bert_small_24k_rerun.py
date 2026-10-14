"""mol_nl BERT small rerun at 1e-3, seed 42, with the evaluation mask held fixed.

The 2026-10-14 order asks for every size to be run again at seed 42 with the validation
set fixed. For BERT that is what bert_small_24k_lr1e3.py already did -- the mask has been
fixed since 2026-09-25 and the grid's seed is 42 -- so this run is a repetition of it
under a name of its own, which is what keeps it out of the finished run's directory.

1e-3 is small's best: 0.0588 as the mean of three seeds, against 0.0687 at 3e-4.

Nothing is changed from bert_small_24k_lr1e3.py.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_small_24k_lr1e3 import *  # noqa: F401,F403
