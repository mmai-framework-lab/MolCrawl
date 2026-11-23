"""rna GPT-2 medium at 1.2e-3, continued from the nine-epoch run to 241,920 iters.

rna-order-2026-11-18b §1-§2; rna-judgment-2026-11-18b 確認事項 1 (case A) and §7.

**lr_decay_iters stays at 120,960, inherited unchanged.** get_lr decides the rate
from the iter number alone, so every iter below 120,960 sees the rate the nine-epoch
run saw at that iter, and every iter above it sees min_lr = 1.2e-4, a tenth of the
peak, which the cosine has already reached at 120,960. There is no step at the join,
and the four sizes stay aligned. Raising lr_decay_iters with max_iters instead would
recompute the whole curve and continue a different run.

**The weights this starts from are at iter 100100, not 120,960.** nanoGPT saves only
on improvement, so ckpt.pt holds the best iter, and all four runs' last evaluation at
120,960 was worse than their best: small 120,100, medium 100,100, large 119,900, xl
119,100 (read off the files on a compute node, job 362444). Resuming restores iter_num
from the checkpoint, so this run re-covers 100,100-120,960 (20,860 iters) at the same rate it saw the first
time. igarashi chose this over re-running to 120,960 first, on 2026-11-18.

**A separate out_dir, holding a copy of ckpt.pt.** The resume path reads
out_dir/ckpt.pt and overwrites it at the first improvement, and the nine-epoch run has
no other copy of those weights. scripts/prepare_rna_extension.py makes the copy.

launch: workflows/rna-gpt2-grid.sbatch with GRID_CONFIG set to this file, once the copy
        is in place. init_from stays "resume" from the base config.
"""

from molcrawl.tasks.pretrain.configs.rna.gpt2_medium_lr1p2e3_9ep import *  # noqa: F401,F403

max_iters = 241920
# lr_decay_iters (120,960), warmup_iters (2,419) and min_lr (1.2e-4) are inherited
# unchanged -- see the header.

# train.py defaults early_stopping to True, and the medium/large/xl bases set it True
# with patience 5, so this has to be stated. §2 of the order disables it. It matters
# most for medium: its best iter is 100,100 and the nine-epoch run then ran 20,860
# iters without improving, so an extension resuming there would stop about 500 iters
# in. workflows/rna-gpt2-grid.sbatch also passes --early_stopping=False, but a value
# that lives only on the command line is invisible to anyone reading the config.
early_stopping = False

import os as _os  # noqa: E402

_runs_root = _os.environ.get("MODEL_OUTPUT_ROOT")
if not _runs_root:
    raise SystemExit(
        "set MODEL_OUTPUT_ROOT to a runs root outside any input tree; "
        "deriving it from LEARNING_SOURCE_DIR puts checkpoints beside the corpus"
    )
out_dir = _os.path.join(_runs_root, "rna-gpt2-medium-lr1p2e3-9ep-ext242k")
tensorboard_dir = out_dir
