"""Throughput arm 1 of 4: as shipped -- fp32, no dataloader workers, no pinning, Arrow.

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

# Nothing overridden. main.py:653-656 reads bf16 / tf32 / dataloader_num_workers
# / dataloader_pin_memory with a False-or-0 default, so the absence of these
# lines is what the production config already means. Stated here so the arm is
# a declaration rather than an omission.
bf16 = False
tf32 = False
dataloader_num_workers = 0
dataloader_pin_memory = False
