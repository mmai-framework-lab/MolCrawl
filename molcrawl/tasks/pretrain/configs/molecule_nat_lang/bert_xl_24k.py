"""mol_nl BERT xl at 24,000 steps, on the schedule the other sizes ran.

The same schedule as bert_{small,medium,large}_24k.py so the sizes are comparable: twice
the 12,000-step grid, warmup at 10%, and the evaluation mask held fixed.

The learning rate is NOT taken from large. The rate that diverges falls with model size
-- 1e-3 at small, 5.5e-4 at medium, 3e-4 at large -- so large's 1.7e-4 is not a safe
starting point for a model 2.6 times its size. The three points that import this file
bracket where the boundary is expected to land.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_xl import *  # noqa: F401,F403

max_steps = 24000
warmup_steps = 2400          # 10%, as at every other size

fixed_eval_mask = True
eval_mask_seed = 42
