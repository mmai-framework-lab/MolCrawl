"""mol_nl BERT small continued past 24,000 steps, to where the validation loss stops
falling (2026-11-16 order §1).

launch: workflows/molnl-bert-ladder.sbatch with EXTEND=1, SEGMENT=2 and
        PREV_JOB=138261 (the completed 24,000-step job), from a COPY of
        bert_small_24k_lr1e3/checkpoint-24000 with scheduler.pt removed. See "The
        copy" below; without it this config's learning rate and warmup are both
        ignored and the run continues at the original peak.

Nothing that defines the model or the data moves: the size, the corpus, the fixed
evaluation mask, the global batch, the seed and the peak this arm was chosen at are the
ones bert_small_24k_lr1e3.py sets, and that file is imported rather than copied. What
changes is the part of the schedule that only exists because the original schedule ended.

The rate is constant because HF decays over max_steps with no separate decay length:
under the default "linear" a larger max_steps does not add steps to the existing curve,
it recomputes the whole curve and continues a different run. constant_with_warmup, 500
steps, to a tenth of this arm's peak -- the shape, the 500 steps and the tenth are what
matsubara chose for the compounds extensions on 2026-10-20, and this follows them rather
than inventing a second convention. The ramp is needed because the original schedule ends
at exactly 0.0 (measured: 4.6296e-06 at step 23,900 and 0.0 at 24,000), so without it the
rate jumps from nothing to 1e-4 at the resume point.

What that costs at the resume point was measured on the compounds extensions, which are
the same shape at the same global batch and the same evaluation cadence: the loss moved
by less than the eval-to-eval noise and was back below its step-15,000 value within 100
to 200 steps (small 0.07119 -> 0.07096 at 15,100; large 0.06807 -> 0.06769 at 15,200).

max_steps is a ceiling, not a target. Early stopping decides where the run ends (§2), and
48,000 is twice the original schedule so that the ceiling is not what the answer turns
out to be. If a run reaches it still improving, that is reported and extended again.

early_stopping has to be stated here. bert_small.py sets it to False -- pretraining ran
its full schedule by design -- and `import *` carries that in, so the trainer default of
True (models/bert/main.py, 2026-11-16) would not apply to this file.

Patience is 90 evaluations, which at log_interval=100 is 9,000 steps without a new best.
10, the default, is too tight for this metric at this cadence: replayed over the compounds
extensions, patience 10 would have stopped three of the four arms 1,100 to 1,800 steps in
(at 16,100 to 16,200 of 30,000) while their best went on falling for another 13,000 steps,
to 0.00242 (small) and 0.00216 (large) below where it stood. The longest run of
evaluations without a new best inside a stretch that was still improving was 82, so 90 is
that with a margin. The cost is the other side of the same number: a run keeps going for
9,000 steps after its last improvement before the stop fires.

The output directory follows the config name, so the 24,000-step run is untouched. Its
ten retained checkpoints hold the values reported on 2026-11-12, and retention keeps the
best ten by score: an extension writing into that directory would prune them.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_small_24k_lr1e3 import *  # noqa: F401,F403

max_steps = 48000            # a ceiling; early stopping decides the end
warmup_steps = 500           # the ramp at the resume point, not a fraction of max_steps
learning_rate = 1e-4         # a tenth of this arm's peak 1e-3
lr_scheduler_type = "constant_with_warmup"
early_stopping = True
early_stopping_patience = 90
