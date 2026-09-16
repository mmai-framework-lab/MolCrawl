"""Throughput arm 3 of 4: bf16 plus four dataloader workers and pinned buffers.

Measurement only -- no run reported as a result uses these. The arms are
cumulative, so arm N carries everything arms 1..N-1 turned on, and each one is
submitted as its own job: within a single job all arms draw the same first rows
(seed 106, global batch 2,560) and 600 MiB fits in page cache, so arm 1 pays the
cold read and the rest do not. That is what made the five-arm run disagree with
itself by 1.80x.

Nothing here is a research value. batch_size x gradient_accumulation_steps x
world_size stays 2,560, so the trajectory is unchanged and only execution moves.
"""

from molcrawl.tasks.pretrain.configs.rna.bert_small import *  # noqa: F401,F403

# Both of these are off in the production config. pin_memory is off because
# main.py:656 defaults it to False, which overrides HuggingFace's own default of
# True -- so this arm adds both, as the instruction reads.
bf16 = True
tf32 = False
dataloader_num_workers = 4
dataloader_pin_memory = True
