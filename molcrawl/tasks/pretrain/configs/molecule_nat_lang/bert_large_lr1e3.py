# mol_nl BERT large — learning-rate grid, point 3 of 3 (lr 1e-3)
# The reference run at this size (87694 + 100709) was 3e-5; it is not part of the grid
# -- different evaluation interval, warmup and seed. It is kept as a reference value.
# (see workflows/molnl-bert-ladder.sbatch and the 2026-09-15 order).
#
# Everything except the learning rate comes from bert_large.py, imported rather than
# copied: nine files each restating max_steps, warmup, the batch split and the seed is
# nine chances for one of them to drift, and a grid whose arms differ in anything else
# does not measure the learning rate.
from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_large import *  # noqa: F401,F403

learning_rate = 1e-3
