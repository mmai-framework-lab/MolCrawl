"""rna GPT-2 medium at learning rate 3e-4, run to 9 epochs.

rna-order-2026-10-14k §1. The 3-epoch grid stopped at 40,320 steps; this takes the
same 17 points to 120,960.

**The schedule is stretched with the run, not just the end point.** Moving max_iters
alone would leave lr_decay_iters at 40,320, and the last 80,640 steps -- two thirds
of the run -- would sit at the floor. train.py's own default carries the note
"should be ~= max_iters per Chinchilla" for exactly this reason.

| | 3 epochs | here |
|---|---|---|
| max_iters | 40,320 | 120,960 |
| lr_decay_iters | 40,320 | 120,960 |
| warmup_iters | 806 (2.0 %) | 2,419 (2.0 %) |
| min_lr | 6e-5 fixed | 3e-4 / 10 |

**min_lr is a tenth of this point's peak, not the fixed 6e-5 the grid used.** That
6e-5 was 5 % of the peak at 1.2e-3 and 80 % of it at 7.5e-5, so the arms decayed by
different factors and the comparison between them carried that as well as the rate.

**Four settings differ from the 3-epoch grid, so the losses are not comparable with
it.** Only the ordering of the rates can be read across the two.

The evaluation reads a fixed set of 3,200 sequences, unchanged from the grid.
"""

from molcrawl.tasks.pretrain.configs.rna.gpt2_medium_lr3e4 import *  # noqa: F401,F403

max_iters = 120960
lr_decay_iters = 120960
warmup_iters = 2419
min_lr = 3e-4 / 10

# The grid's three lower rates resampled their validation rows every eval; only the
# 1.2e-3 points were fixed. All seventeen are fixed here, so the curves can be read
# at the width of the differences between them.
deterministic_val_eval = True

seed = 42

import os as _os  # noqa: E402

_runs_root = _os.environ.get("MODEL_OUTPUT_ROOT")
if not _runs_root:
    raise SystemExit(
        "set MODEL_OUTPUT_ROOT to a runs root outside any input tree; "
        "deriving it from LEARNING_SOURCE_DIR puts checkpoints beside the corpus"
    )
out_dir = _os.path.join(_runs_root, "rna-gpt2-medium-lr3e4-9ep")
tensorboard_dir = out_dir
