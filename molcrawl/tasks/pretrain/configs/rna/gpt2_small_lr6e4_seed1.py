"""rna GPT-2 small at learning rate 6e-4, seed 1.

rna-order-2026-10-01 §2. The grid's differences between learning rates have come
down to 0.0051 at small, 0.0034 at medium and 0.0014 at large, measured on the
whole validation split. Whether any of those is a ranking depends on how far the
same arm moves when only the seed changes, and that has never been measured for
rna. On molecule_nat_lang's GPT-2 the spread between seeds was 0.0119 to 0.0333 --
larger than all three of rna's differences.

Seeds 1 and 17 are the pair the seed policy names for a variance run; the grid's
own 42 is the third member, so each rate ends up with three.

**No data_seed.** One seed drives both the weights and the batch order
(train.py:429-432), and train.py refuses a config that declares a second one.

Everything else comes from the grid's own config for this rate, imported rather
than copied, so max_iters, lr_decay_iters, warmup_iters, block_size and the batch
shape cannot drift from the arm this is measuring the spread of.
"""

from molcrawl.tasks.pretrain.configs.rna.gpt2_small_lr6e4 import *  # noqa: F401,F403

seed = 1

import os as _os  # noqa: E402

_runs_root = _os.environ.get("MODEL_OUTPUT_ROOT")
if not _runs_root:
    raise SystemExit(
        "set MODEL_OUTPUT_ROOT to a runs root outside any input tree; "
        "deriving it from LEARNING_SOURCE_DIR puts checkpoints beside the corpus"
    )
out_dir = _os.path.join(_runs_root, "rna-gpt2-small-lr6e4-seed1")
tensorboard_dir = out_dir
