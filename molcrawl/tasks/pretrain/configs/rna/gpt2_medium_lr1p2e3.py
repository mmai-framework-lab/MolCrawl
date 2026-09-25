"""rna GPT-2: medium at learning rate 1.2e-3.

The point above the grid. 6e-4 was the grid's top rate and came out best at
every one of the four sizes, on validation loss and on the cell-type probe
alike, so the grid never measured what happens above it. The grid steps by a
factor of two (1.5e-4 / 3e-4 / 6e-4); this is the next mark.

All four sizes carry this point because the rate at which a run collapses falls
as the model grows: in the rna BERT grid, small and medium collapsed at 1e-3
while large already collapsed at 3e-4. A result from one size does not transfer
to another.

Everything but the learning rate, the seed and the output directory comes from
the base config, so max_iters, lr_decay_iters, warmup_iters, block_size and the
batch shape stay identical to the grid this point extends.
"""

from molcrawl.tasks.pretrain.configs.rna.gpt2_medium import *  # noqa: F401,F403

learning_rate = 1.2e-3

# 42, as in every point of the grid, so initialisation does not move with the
# point being compared.
seed = 42

# Named from this config. MODEL_OUTPUT_ROOT keeps the checkpoints outside the
# tree the run reads from.
import os as _os  # noqa: E402

_runs_root = _os.environ.get("MODEL_OUTPUT_ROOT")
if not _runs_root:
    raise SystemExit(
        "set MODEL_OUTPUT_ROOT to a runs root outside any input tree; "
        "deriving it from LEARNING_SOURCE_DIR puts checkpoints beside the corpus"
    )
out_dir = _os.path.join(_runs_root, "rna-gpt2-medium-lr1p2e3")
tensorboard_dir = out_dir

# The shape the grid ran at. The effective batch is 2,560 either way, so this
# changes residency, not training.
batch_size = 16
gradient_accumulation_steps = 160
expected_global_batch = 2560

dtype = "bfloat16"
