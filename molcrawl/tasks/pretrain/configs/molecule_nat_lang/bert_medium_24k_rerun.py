"""mol_nl BERT medium rerun at 3e-4, seed 42, with the evaluation mask held fixed.

The 2026-10-14 order asks for every size to be run again at seed 42 with the validation
set fixed. For BERT that is what bert_medium_24k_lr3e4.py already did -- the mask has been
fixed since 2026-09-25 and the grid's seed is 42 -- so this run is a repetition of it
under a name of its own, which is what keeps it out of the finished run's directory.

3e-4 rather than 4.1e-4, which reads lower (0.0559) but diverged on one of its three
seeds. The 2026-10-05 order settled that the stable rate stands for the size.

Nothing is changed from bert_medium_24k_lr3e4.py.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_medium_24k_lr3e4 import *  # noqa: F401,F403
