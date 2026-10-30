# rna BERT small lr 3e-4 — re-run of steps 115,301..119,200 to produce the checkpoint
# at the eval_loss_mask minimum, which the original run evaluated but never saved.
#
# all-checkpoint-metric-judgment-2026-10-30-v3 §9. rna-small-lr3e4 saved
# checkpoint-115300 (mask 2.695726) and HF marked it best; the minimum over the whole
# log_history is step 119,200 (2.692825). 3,900 steps of re-run, against medium's 800,
# because this run's minimum sits further from the save grid.
#
# NOT LAUNCHED. Written ahead of the decision; the output directory has to be seeded
# with a copy of checkpoint-115300 first. See workflows/rna-ck-sensitivity-prepare.sbatch.
#
# Nothing about the run changes: max_steps stays 120,960 for the schedule, stop_at_step
# ends the loop at 119,200 with an evaluation and a save, seed is the grid's 42.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_small_lr3e4 import *  # noqa: F401,F403

stop_at_step = 119200
model_path = get_bert_output_path("rna", "small") + "-lr3e4-ck119200"
