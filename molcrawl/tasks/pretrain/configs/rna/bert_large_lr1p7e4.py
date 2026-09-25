# rna BERT large -- filling in the grid, learning rate 1.7e-4.
# rna-order-2026-09-25 §7.2. The first grid left every size with its best rate at an
# end of the range: small and medium were best at 3e-4 and collapsed at 1e-3, large was
# best at 1e-4 and collapsed at 3e-4. A best that sits at an end is not bracketed, so
# the range between the rate that worked and the rate that collapsed is divided into
# four in the logarithm and the three interior marks are placed here. An even step
# means that wherever the collapse begins, it is located to within that one step.
#
# large: 1e-4 to 3e-4, step 1.32x.
#
# Everything else comes from bert_large.py, imported rather than copied, so these points
# land on the same grid as the first nine: max_steps 120,960 and warmup_steps 12,096
# (9 epochs, 10 %), 8 x 80 on 4 GPUs, bf16 with four dataloader workers and pinned
# buffers, document masking with boundary id 0.
#
# seed is 42 as in the first nine. model_path is named after this config so no two
# rates of a size share a directory; it resolves under MODEL_OUTPUT_ROOT, and with that
# unset main.py's output guard stops the run before its first step.
from molcrawl.core.paths import get_bert_output_path
from molcrawl.tasks.pretrain.configs.rna.bert_large import *  # noqa: F401,F403

learning_rate = 1.7e-4
seed = 42
model_path = get_bert_output_path("rna", "large") + "-lr1p7e4"
