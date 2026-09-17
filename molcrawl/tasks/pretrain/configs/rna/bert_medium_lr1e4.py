# rna BERT medium -- learning-rate grid, point 1 of 3 (lr 1e-4).
# all-bert-order-2026-09-17 §3.2. The grid is 3 sizes x (1e-4, 3e-4, 1e-3), seed 42 in all nine.
#
# Everything else comes from bert_medium.py, imported rather than copied, the way the
# molecule_nat_lang and protein_sequence grids are written: max_steps 120,960 and
# warmup_steps 12,096 (9 epochs, 10 %), 8 x 80 on 4 GPUs, bf16 with four dataloader
# workers and pinned buffers, document masking with boundary id 0. A grid whose arms
# differ in anything but the learning rate does not measure the learning rate.
#
# Three names are set here. learning_rate is the point on the grid. seed is 42 in all
# nine, where the bases carry their own sequential seeds (106 / 105 / 104). model_path
# is named after this config so the three rates of a size never share a directory;
# it resolves under MODEL_OUTPUT_ROOT, and with that unset main.py's output guard stops
# the run before its first step rather than writing into the input tree.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_medium import *  # noqa: F401,F403

learning_rate = 1e-4
seed = 42
model_path = get_bert_output_path("rna", "medium") + "-lr1e4"
