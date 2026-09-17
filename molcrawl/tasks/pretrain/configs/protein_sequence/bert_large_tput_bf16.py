"""Throughput measurement: the protein_sequence large base config with bf16 on.

Measurement only. The base already carries four dataloader workers and pinned
buffers; this adds bf16, so with the base itself the pair gives the fp32 and bf16
values of the same condition (all-bert-continuation-verdict-2026-09-17 §2.2).
The bf16 value is the one estimates are built on.
"""

from molcrawl.tasks.pretrain.configs.protein_sequence.bert_large import *  # noqa: F401,F403

bf16 = True
