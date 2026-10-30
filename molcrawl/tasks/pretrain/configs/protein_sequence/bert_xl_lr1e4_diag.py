# protein BERT xl diagnostic — learning rate. lr 1e-4, stopped at step 2,000.
#
# all-checkpoint-metric-judgment-2026-10-30-v3 §6 (arm ア). The xl grid spans 5.5e-6
# to 3.4e-5, a factor of 6.2, chosen by extrapolating the collapse boundary downwards
# from large (protein-grid-followup-order-2026-09-30 §4 predicted xl's near 8e-6). It
# never reached the band the smaller sizes are best in: medium's best is 1e-4 (mask
# 2.4938) and small's 3e-4 (2.5072). The one xl arm near it, 3.4e-5, did not explode:
# it settled at 2.894, the amino-acid unigram baseline, which is what a run that has
# learned only the marginal distribution reads. So "the grid sat below the rate this
# size needs" is still open, and 4.2x the top of the grid tests it.
#
# NOT LAUNCHED. Written ahead of the decision; §6 asks for the arms to be described
# before they are submitted.
#
# Read-out is the TRAIN loss at step 400 and 2,000 against the same baselines as the
# fp32 arm (xl 2.8485 / 2.8228, large 2.6559 / 2.6220). Everything else is the grid
# arm's: adam_beta2 0.999, warmup 3,353, max_steps 33,531, 8 x 80 x 4 = 2,560, bf16.
# stop_at_step ends the loop at 2,000 with the schedule left alone, so the learning
# rate at step 2,000 is 1e-4 * 2000/3353 = 5.96e-5 -- still on the warmup ramp, as it
# is for every other arm at that step.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_xl_lr2p4e5 import *  # noqa: F401,F403

learning_rate = 1e-4        # the one thing under test (grid top was 3.4e-5)
stop_at_step = 2000
model_path = get_bert_output_path("protein_sequence", "xl") + "-lr1e4-diag"
