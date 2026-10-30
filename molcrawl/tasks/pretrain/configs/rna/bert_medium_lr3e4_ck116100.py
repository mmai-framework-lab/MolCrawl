# rna BERT medium lr 3e-4 — re-run of steps 115,301..116,100 to produce the checkpoint
# at the eval_loss_mask minimum, which the original run evaluated but never saved.
#
# all-checkpoint-metric-judgment-2026-10-30-v3 §9. rna-medium-lr3e4 saved
# checkpoint-115300 (mask 2.638232) and HF marked it best, but the minimum over the
# whole log_history is step 116,100 (2.636773) -- 100 steps off the save grid, and the
# only downstream probe of this run read 115,300. The 3-point measurement asks for that
# step's weights, so they have to exist first.
#
# NOT LAUNCHED. Written ahead of the decision. The output directory has to be seeded
# with a copy of checkpoint-115300 before this config can do anything; see
# workflows/rna-ck-sensitivity-prepare.sbatch and workflows/submit-rna-ck-sensitivity.sh.
#
# Nothing about the run changes. max_steps stays 120,960 so the learning-rate schedule
# is the one the original decayed over, and stop_at_step ends the loop at 116,100 with
# an evaluation and a save (models/bert/_stop_at_step). seed is the grid's 42, which is
# also what the original run actually used, so the resumed data order is the original's
# -- including on fix/bert-seed-reaches-transformers, where a config's seed finally
# takes effect.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_medium_lr3e4 import *  # noqa: F401,F403

stop_at_step = 116100
model_path = get_bert_output_path("rna", "medium") + "-lr3e4-ck116100"
