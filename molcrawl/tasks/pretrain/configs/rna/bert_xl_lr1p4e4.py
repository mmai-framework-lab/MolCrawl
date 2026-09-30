# rna BERT xl -- learning rate 1.4e-4. rna-followup-order-2026-09-30 §5.
# The expected boundary, from large's 2.3e-4-to-3e-4 divided by 1.8.
#
# The three points are 1e-4 / 1.4e-4 / 2e-4, a step of 1.4. The middle one is where
# the boundary is expected: large collapses between 2.3e-4 and 3e-4, and in rna the
# boundary falls by about 1.8 for one step of size, which puts xl near 1.4e-4.
#
# That 1.8 is rna's own. In protein_sequence the same step is about 3, so the ratio
# is not carried between modalities; it is taken from this modality's large.
#
# Everything else comes from bert_xl.py: max_steps 120,960, warmup_steps 12,096,
# 8 x 80 on 4 GPUs for an effective 2,560, bf16, four workers, pinned buffers,
# document masking with boundary id 0. seed 42 as in every point of the rna grid;
# main.py sets data_seed from it, so the shuffling matches too.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_xl import *  # noqa: F401,F403

learning_rate = 1.4e-4
seed = 42
model_path = get_bert_output_path("rna", "xl") + "-lr1p4e4"
