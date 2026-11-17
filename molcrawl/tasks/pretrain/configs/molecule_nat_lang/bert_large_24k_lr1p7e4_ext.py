"""mol_nl BERT large continued past 24,000 steps, to where the validation loss stops
falling (2026-11-16 order §1).

launch: workflows/molnl-bert-ladder.sbatch with EXTEND=1, SEGMENT=2 and
        PREV_JOB=138279 (the completed 24,000-step job), from a COPY of
        bert_large_24k_lr1p7e4/checkpoint-24000 with scheduler.pt removed. See "The
        copy" in bert_small_24k_lr1e3_ext.py; without it this config's learning rate
        and warmup are both ignored and the run continues at the original peak.

Nothing that defines the model or the data moves: the size, the corpus, the fixed
evaluation mask, the global batch, the seed and the peak this arm was chosen at are the
ones bert_large_24k_lr1p7e4.py sets, and that file is imported rather than copied.
What changes is the part of the schedule that only exists because the original schedule
ended.

The rate is constant because HF decays over max_steps with no separate decay length:
under the default "linear" a larger max_steps does not add steps to the existing curve,
it recomputes the whole curve and continues a different run. constant_with_warmup, 500
steps, to 1.7e-5 -- a tenth of this arm's peak 1.7e-4. The shape, the 500 steps and
the tenth are what matsubara chose for the compounds extensions on 2026-10-20, and this
follows them rather than inventing a second convention. The ramp is needed because the
original schedule ends at exactly 0.0 (measured: 7.8704e-07 at step 23,900 and 0.0 at
24,000), so without it the rate jumps from nothing to 1.7e-5 at the resume point.

What that costs at the resume point was measured on the compounds extensions, which are
the same shape at the same global batch and the same evaluation cadence: the loss moved
by less than the eval-to-eval noise and was back below its step-15,000 value within 100
to 200 steps.

max_steps is a ceiling, not a target. Early stopping decides where the run ends (§2), and
48,000 is twice the original schedule so that the ceiling is not what the answer turns
out to be. If a run reaches it still improving, that is reported and extended again.

early_stopping has to be stated here: bert_large.py sets it to False and `import *`
carries that in, so the trainer default of True would not apply to this file. Patience 90
evaluations = 9,000 steps without a new best; the measurement behind that number is in
bert_small_24k_lr1e3_ext.py.

The output directory follows the config name, so the 24,000-step run is untouched. Its
ten retained checkpoints hold the values reported on 2026-11-12, and retention keeps the
best ten by score: an extension writing into that directory would prune them.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_large_24k_lr1p7e4 import *  # noqa: F401,F403

max_steps = 48000            # a ceiling; early stopping decides the end
warmup_steps = 500           # the ramp at the resume point, not a fraction of max_steps
learning_rate = 1.7e-5         # a tenth of this arm's peak 1.7e-4
lr_scheduler_type = "constant_with_warmup"
early_stopping = True
early_stopping_patience = 90
