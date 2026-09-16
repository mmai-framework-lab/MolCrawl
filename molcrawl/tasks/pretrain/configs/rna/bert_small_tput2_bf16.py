"""Throughput arm 2 of 4: bf16 mixed precision.

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

# The measured rate at 18.8 s/step is 29.7 TFLOP/s once the 2.47 s read is taken
# out; genome's BERT small, same hidden size and layer count with bf16 on, runs
# at 253 TFLOP/s. On GB200 the first is an fp32 rate and the second cannot be
# one. This arm is the check on that reading.
bf16 = True
tf32 = False
dataloader_num_workers = 0
dataloader_pin_memory = False
