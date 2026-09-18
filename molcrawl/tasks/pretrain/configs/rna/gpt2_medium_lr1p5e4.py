"""rna GPT-2 grid: medium at learning rate 1.5e-4.

One point of the 4 x 3 (+1) grid. The four earlier runs each used a different
learning rate, so the differences between them carried both the size and the
rate; this holds the rate to the same three points at every size, and adds
7.5e-5 at xl the way protein's grid does.

Everything but the learning rate, the seed and the output directory comes from
the base config, so max_iters, lr_decay_iters, warmup_iters, block_size and the
batch shape cannot drift between the points of a grid that is meant to differ in
one thing.
"""

from molcrawl.tasks.pretrain.configs.rna.gpt2_medium import *  # noqa: F401,F403

learning_rate = 1.5e-4

# 42 across the grid, not the per-config number assigned on 2026-08-03. A seed
# that moves with the point puts initialisation differences inside the
# size-and-rate comparison the grid exists to make.
seed = 42

# Named from this config, so the three points of a size cannot write to one
# place. MODEL_OUTPUT_ROOT keeps them outside the tree the run reads from.
import os as _os  # noqa: E402

_runs_root = _os.environ.get("MODEL_OUTPUT_ROOT")
if not _runs_root:
    raise SystemExit(
        "set MODEL_OUTPUT_ROOT to a runs root outside any input tree; "
        "deriving it from LEARNING_SOURCE_DIR puts checkpoints beside the corpus"
    )
out_dir = _os.path.join(_runs_root, "rna-gpt2-medium-lr1p5e4")
tensorboard_dir = out_dir

# Both sizes were checked at 16 x 160 on a GB200 (jobs 123495 / 123496, three
# steps each): the base configs lower the micro-batch for large and xl "to stay
# memory-safe", which held on the earlier hardware and does not here. The
# effective batch is 2,560 either way, so this changes residency, not training.
batch_size = 16
gradient_accumulation_steps = 160
expected_global_batch = 2560

# train.py picks bfloat16 when the device supports it, which a GB200 does.
# Stated here so the run does not depend on what the hardware happens to offer.
dtype = "bfloat16"
