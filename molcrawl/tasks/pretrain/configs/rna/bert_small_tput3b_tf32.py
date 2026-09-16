"""Throughput arm 3b of 4: tf32 alone, without bf16.

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

# Not one of the four arms in the instruction. It is here because the port
# already exists (main.py:654), it costs one line, and if fp32 is what the step
# is spending on then this separates "the matmuls were not using the tensor
# cores at all" from "bf16 also halves the traffic". Runs only if asked for.
bf16 = False
tf32 = True
dataloader_num_workers = 0
dataloader_pin_memory = False
