# mol_nl BERT medium — learning-rate grid, point 2 of 3 (lr 3e-4)
# The reference run at this size (87693 + 100708) was 1e-4; it is not part of the grid
# -- different evaluation interval, warmup and seed. It is kept as a reference value.
# (see workflows/molnl-bert-ladder.sbatch and the 2026-09-15 order).
#
# Everything except the learning rate comes from bert_medium.py, imported rather than
# copied: nine files each restating max_steps, warmup, the batch split and the seed is
# nine chances for one of them to drift, and a grid whose arms differ in anything else
# does not measure the learning rate.
from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_medium import *  # noqa: F401,F403

learning_rate = 3e-4
