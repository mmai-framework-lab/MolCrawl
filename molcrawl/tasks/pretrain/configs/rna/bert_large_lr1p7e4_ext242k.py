# rna BERT large -- the 1.7e-4 arm continued from 120,960 steps to 241,920.
# rna-order-2026-11-18b §1-§2; rna-judgment-2026-11-18b 確認事項 1, 2 and §7.
#
# launch: workflows/bert-grid.sbatch with EXTEND=1, resuming from the COPY of
#         checkpoint-120960 that scripts/prepare_rna_extension.py writes into this
#         config's own output directory. Resuming from the original directory would
#         both overwrite the 9-epoch record and ignore the two rates below.
#
# The rate is constant because HF has no decay length separate from max_steps: under
# the default "linear" a larger max_steps does not add steps to the existing curve,
# it recomputes the whole curve and continues a different run. constant_with_warmup
# at a tenth of this arm's peak, ramped over 500 steps, is the shape igarashi
# approved on 2026-11-18. The 9-epoch schedule ends at exactly 0.0 -- read off
# checkpoint-120960/scheduler.pt, whose _last_lr is [0.0, 0.0] -- so without the ramp
# the rate steps from nothing to the new value at the resume point.
#
# 241,920 is 18 epochs: the 9-epoch schedule divides into 13,440 steps per epoch.
#
# A separate output directory, so the 9-epoch run is untouched. Checkpoint retention
# keeps the best ten by eval_loss_mask, and BestNCheckpointRetention counts a resumed
# run's earlier steps as its own, so running in place would put this arm's reported
# 9-epoch checkpoint (checkpoint-115300, eval_loss_mask 2.6373) on the prune list.

from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_large_lr1p7e4 import *  # noqa: F401,F403

max_steps = 241920
warmup_steps = 500  # the ramp at the resume point, not a fraction of max_steps
learning_rate = 1.7e-5  # a tenth of this arm's peak 1.7e-4
lr_scheduler_type = "constant_with_warmup"
early_stopping = False  # the order asks for it explicitly; the base sets it too
model_path = get_bert_output_path("rna", "large") + "-lr1p7e4-ext242k"

# Collapse detection, approved 2026-11-18. A collapsed run never recovers, so stop it
# rather than spend the added 120,960 steps at the degenerate level -- about 2,970 GPU
# hours on xl. The detector reads eval_loss_mask, the metric the order ranks on.
#
# 9.0 is measured, not derived. Across the fourteen rna arms that ended collapsed, the
# LOWEST sustained level is 9.3737 (large 3e-4, median of its last 2,000 steps, range
# 9.3704-9.3753), so 9.0 catches every one of them with 0.37 to spare. It also sits
# 5.7 above the worst value this extension starts from (xl 3.3168).
#
# The obvious derivation would be wrong, and only xl shows why. Scaling the unigram
# floor by the non-copy fraction gives 0.9 x 9.3710 = 8.4296, but a HEALTHY xl 2.5e-5
# run is still above 8.5 at step 20,800 and above 8.0 at step 59,000 -- far past the
# 5,000-step grace period. Any threshold below 9.0 would stop a run that was working.
#
# grace_steps is compared against the absolute global_step, and this run resumes at
# 120,960, so the detector is live from its first evaluation rather than waiting out a
# plateau that has already been crossed. patience 5 at log_interval 100 means 500
# steps above the line before stopping.
degenerate_loss_threshold = 9.0
degenerate_patience = 5
degenerate_grace_steps = 5000
