"""mol_nl GPT-2 medium at 1,500 iterations, learning rate 6e-4, seed 1.

One of the repeat runs the 2026-09-30 order asks for. The envelope over model sizes is
read as "flat, and reversing at the largest size", and the differences it rests on are
0.0022 for BERT and 0.017 for GPT-2 -- against a run-to-run spread that has never been
measured here. Two more seeds per size at the rate that did best for it is what turns
that reading into something that can be stated or withdrawn.

Everything except the seed comes from gpt2_medium_1500_lr6e4, imported rather than copied.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_medium_1500_lr6e4 import *  # noqa: F401,F403

seed = 1
