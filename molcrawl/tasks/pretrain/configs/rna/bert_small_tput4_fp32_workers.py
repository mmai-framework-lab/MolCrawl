"""Throughput arm 4: fp32 with four dataloader workers and pinned buffers.

This is the combination that was approved for production and the one the nine-run
estimate has to come from. Arm 3 measured 3.812 s/step with bf16 also on, and bf16
was not approved: on its own it is 0.94x, it changes numerics rather than placement,
and turning it on for rna alone would leave the other four modalities in fp32.

Runs on a seed no other arm has used. Every arm so far shares seed 106, so each one
draws the same rows from the head of the same permutation; one arm per job and one
arm at a time do not fix that, because a later arm landing on a node an earlier arm
ran on finds those rows in RAM. Rows are all 1,024 tokens, so which rows are drawn
does not change what a step costs -- only whether the read is warm does.
"""

from molcrawl.tasks.pretrain.configs.rna.bert_small import *  # noqa: F401,F403

bf16 = False
tf32 = False
dataloader_num_workers = 4
dataloader_pin_memory = True
