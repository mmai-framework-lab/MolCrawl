# protein BERT xl diagnostic — precision. lr 2.4e-5, bf16 off, stopped at step 2,000.
#
# all-checkpoint-metric-judgment-2026-10-30-v3 §6 (arm ア). The six xl arms stop
# improving at step 400 and gain 0.006 over the remaining 33,100 steps, while large at
# the same point on its warmup is already at train 2.656 (xl 2.849). §5 ruled out the
# first suspicion: every xl arm ran bf16=True, as large did, so precision is not an
# asymmetry between the two sizes -- but 48 layers is where bf16 accumulation in the
# residual stream would start to bite, and large's 36 do not. This arm is the only
# change that can be made without touching code.
#
# NOT LAUNCHED. Written ahead of the decision so the launch is one command. §6 asks
# for the arms to be described before they are submitted.
#
# Read-out is the TRAIN loss at step 400 and 2,000, not eval_loss_mask: at step 2,000
# large's mask is 2.8176 and xl's 2.8264, so mask does not separate the sizes this
# early. Train does, by 0.20. Baselines from the completed runs:
#   xl-lr2p4e5     step 400 2.8485, step 2,000 2.8228
#   large-lr3p5e5  step 400 2.6559, step 2,000 2.6220
#
# fp32 doubles the activation memory, so the micro-batch halves and the accumulation
# doubles: 4 x 160 x 4 GPUs = 2,560, the batch the schedule was derived from, kept
# identical so the step count means the same thing. max_steps and warmup are the
# base's 33,531 / 3,353; stop_at_step ends the loop at 2,000 without changing the
# schedule the learning rate decays over (models/bert/_stop_at_step).
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.protein_sequence.bert_xl_lr2p4e5 import *  # noqa: F401,F403

bf16 = False                # the one thing under test
batch_size = 4              # fp32 activations; 4 x 160 x 4 = 2,560 unchanged
gradient_accumulation_steps = 5 * 32
stop_at_step = 2000
model_path = get_bert_output_path("protein_sequence", "xl") + "-lr2p4e5-diag-fp32"
