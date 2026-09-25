"""mol_nl BERT large at 24,000 steps -- the schedule the 2026-09-25 order asks for.

Twice the 12,000-step grid, with warmup kept at 10%. The 12,000-step runs stay where they
are; this writes to its own directories, one per learning rate.

Whether 12,000 steps was enough cannot be read off those runs: the learning rate decays to
zero at max_steps, so the last evaluation being the lowest is what a decayed schedule does
whether or not the budget was short. Running the same points at twice the length is what
answers it -- the six points this shares with the 12,000-step grid (1e-4, 3e-4, and 1e-3 at
small) are there to be read against their shorter twins.

Evaluation masks a fixed set of positions here, which the 12,000-step runs did not do. Values
from the two schedules are comparable in trend but not to the last digit: the shorter runs
redrew the mask at every evaluation.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_large import *  # noqa: F401,F403

max_steps = 24000
warmup_steps = 2400          # 10%, as at 12,000 steps

# The masked positions are the same at every evaluation, so two evaluations of one model
# differ by the model alone. On the 12,000-step grid the redraw moved the number by 0.0074
# to 0.0159 over the last ten evaluations, against a 0.0062 gap between the best small and
# the best medium arm. Training masks are untouched: no gradient changes.
fixed_eval_mask = True
eval_mask_seed = 42
