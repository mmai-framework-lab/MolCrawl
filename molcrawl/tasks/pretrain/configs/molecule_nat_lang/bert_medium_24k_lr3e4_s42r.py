"""mol_nl BERT medium at 24,000 steps, learning rate 3e-4, seed 42.

medium is the one size whose representative rate carries a single seed. The three-seed
sets are small at 1e-3 and large at 1.7e-4; medium's three sit at 4.1e-4, which is not
the rate that stands for it, and one of those three diverged at step 13,000. That is why
the 2026-10-19 report leaves medium without a seed spread, and why the step from medium
to large cannot be stated with one.

All three seeds are run together rather than only the two that are missing. PR #307 made
eval_loss_mask the sum over ranks where it had been rank 0's quarter shard, so the
existing seed-42 run is measured under the other definition; a spread taken across the
two would measure the definition rather than the seeds.

This one repeats the conditions of bert_medium_24k_lr3e4 under a name of its own, so that
the finished run stays where it is and the reduced metric is measured in a fresh tree.

Everything except the seed comes from bert_medium_24k_lr3e4, imported rather than copied.

Launched 2026-10-22 as job 199851.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_medium_24k_lr3e4 import *  # noqa: F401,F403

seed = 42
