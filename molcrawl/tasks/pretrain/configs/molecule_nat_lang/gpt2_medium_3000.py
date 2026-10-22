"""mol_nl GPT-2 medium at 6e-4, over 3,000 iterations instead of 1,500.

1,500 iterations is 12.07 epochs of this corpus, against the 193.1 the BERT grid runs,
and the validation loss is still descending when the schedule ends. The sizes cannot be
ordered at that budget either: the gaps between them are 0.002 to 0.006 where the spread
across seeds 42 / 1 / 17 of one arm is 0.0153 to 0.0322.

This doubles the schedule and nothing else. It is a fresh run rather than a continuation
because lr_decay_iters moves with max_iters -- a rate that decays over a different length
is a different experiment, not more of the same one -- and the arm's own 1,500-iteration
run stays where it is for the comparison.

6e-4 is the rate this size did best at in the grid (window mean 0.5953 at seed 42).
Everything else, including the fixed validation set, comes from
gpt2_medium_1500_fixedval.py.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_medium_1500_fixedval import *  # noqa: F401,F403

max_iters = 3000
lr_decay_iters = 3000        # decay over the schedule that is actually run
warmup_iters = 60            # 2% of max_iters, the convention the base config states
