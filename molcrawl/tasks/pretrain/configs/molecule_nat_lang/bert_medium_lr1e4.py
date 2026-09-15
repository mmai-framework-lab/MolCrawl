# mol_nl BERT medium — learning-rate grid, point 1 of 3 (lr 1e-4)
# Run 87693 + 100708 sat at 1e-4 but is NOT this point: it evaluated every 250 steps
# against this grid's 100, ran warmup 200 against 1,200, and used seed {59:'59',55:'55',54:'54'}
# (see workflows/molnl-bert-ladder.sbatch and the 2026-09-15 order).
#
# Everything except the learning rate comes from bert_medium.py, imported rather than
# copied: nine files each restating max_steps, warmup, the batch split and the seed is
# nine chances for one of them to drift, and a grid whose arms differ in anything else
# does not measure the learning rate.
from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_medium import *  # noqa: F401,F403

learning_rate = 1e-4
