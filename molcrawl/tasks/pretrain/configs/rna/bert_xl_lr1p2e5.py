# rna BERT xl -- learning rate 1.2e-5. rna-order-2026-10-14k §6.
# Near the effective rate the first three points topped out at.
#
# The first three xl points (1e-4 / 1.4e-4 / 2e-4) never learned: their best values
# were 9.07 to 9.37 against a baseline of 9.372, and by step 16,000 they sat at 10.6
# where the other sizes reach 2.7 long before. 5e-5 did descend -- 8.64 at 7,200
# steps, still improving -- so the boundary is below 1e-4 and these three walk down
# from there.
#
# Everything else comes from bert_xl.py, so the comparison is against the four
# points already run and not against a differently configured run. seed 42, and no
# data_seed: main.py refuses a config that declares a second seed.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_xl import *  # noqa: F401,F403

learning_rate = 1.2e-5
seed = 42
model_path = get_bert_output_path("rna", "xl") + "-lr1p2e5"
