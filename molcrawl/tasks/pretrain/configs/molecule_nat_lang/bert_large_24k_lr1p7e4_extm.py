"""mol_nl BERT large continued past 24,000 steps, with the optimizer's moments kept.

launch: workflows/molnl-bert-ladder.sbatch with EXTEND=1, SEGMENT=2 and
        PREV_JOB=356045, from a copy of bert_large_24k_lr1p7e4/checkpoint-24000
        whose scheduler.pt has been re-pointed at this config
        (scripts/extension_scheduler_rewrite.py). The launcher's gate checks the
        file's contents and refuses an unprepared copy.

Why this exists beside bert_large_24k_lr1p7e4_ext.py. That run took the same
schedule from the same checkpoint, but its copy had scheduler.pt REMOVED, and
transformers guards the optimizer load and the scheduler load with one condition on
that file: the Adam moments were never restored. Measured afterwards on its own
checkpoints -- the Adam step counter reads 100 at step 24,100, against 24,000 in the
checkpoint it resumed from. So that run was the weights re-optimized with a fresh Adam,
not the run continued, and what it ended up doing (33,000 で早期停止) cannot be separated from the
moments restarting. Re-pointing the file instead keeps the rate this config asks for AND
the moments. The earlier run stays where it is; this one writes to its own directory.

Everything that defines the model and the data is bert_large_24k_lr1p7e4.py's,
imported rather than copied: the size, the corpus, the fixed evaluation mask, the global
batch, the seed and the peak this arm was chosen at.

The rate past the original schedule is a constant tenth of that peak, 1.7e-5, with a
500-step ramp, which is the shape matsubara chose for the compounds extensions on
2026-10-20. The ramp is needed because the original schedule ends at exactly 0.0.

max_steps is 72,000 rather than the 48,000 the first extension used, so that early
stopping decides where this run ends rather than the ceiling -- which is what the
2026-11-17 order asks for, and which 48,000 did not deliver for small: it reached the
ceiling with its minimum at step 43,700 and the patience counter at 43 of 90.

Patience is 90 evaluations = 9,000 steps without a new best; the measurement behind that
number is in bert_small_24k_lr1e3_ext.py. early_stopping is stated here because
bert_large.py sets it to False and `import *` carries that in.

Two numbers this run is read against. The run it continues has its best at 0.057246
(step 23,800; the resume point is its last step, 24,000). The moment-less extension
of the same checkpoint never improved on it -- its own minimum was 0.057853.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_large_24k_lr1p7e4 import *  # noqa: F401,F403

max_steps = 72000            # a ceiling; early stopping decides the end
warmup_steps = 500           # the ramp at the resume point, not a fraction of max_steps
learning_rate = 1.7e-5         # a tenth of this arm's peak 1.7e-4
lr_scheduler_type = "constant_with_warmup"
early_stopping = True
early_stopping_patience = 90
