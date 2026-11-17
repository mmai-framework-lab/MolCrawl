"""mol_nl GPT-2 medium continued past 1,500 iterations, to where the validation loss
stops falling (2026-11-16 order §1).

launch: workflows/molnl-gpt2-extend.sbatch, from a COPY of
        gpt2_medium_1500_lr6e4/checkpoint-1500 and that run's logging CSV. The
        copy is what keeps the 1,500-iteration run's checkpoints out of this run's
        retention ranking, and the CSV is what keeps the earlier evaluations in it
        (train.py, eval_history).

Nothing that defines the model or the data moves: the size, the corpus, the global batch
of 2,560 sequences of 1,024 tokens, the seed and the peak rate this arm was chosen at are
the ones gpt2_medium_1500_lr6e4.py sets, and that file is imported rather than
copied.

lr_decay_iters stays at 1,500 and that is the whole of the rate decision. get_lr returns
min_lr for every iteration past lr_decay_iters (models/gpt2/train.py), and the cosine is
already at min_lr when it reaches 1,500, so the extension runs at a constant 6e-5 --
a tenth of the peak 6e-4 -- with no step change at the resume point and no warmup to
add. Moving lr_decay_iters with max_iters instead would recompute the rate at every
iteration from the first, which is a different experiment and not a continuation of this
one: that is why gpt2_medium_3000.py is a fresh run rather than a resume.

max_iters is a ceiling, not a target. Early stopping decides where the run ends (§2).
3,000 is the length of the existing 3,000-iteration runs, and those turned at 1,400 to
1,600 iterations and rose steeply after -- on a fixed validation subset, so their numbers
are not this arm's, but the shape says a minimum within a few hundred iterations of here
is the expected outcome, not a long climb. Replayed over those four runs, patience 10
fires 300 to 500 iterations past the minimum and the minimum itself is kept: nanoGPT
writes a checkpoint at every improvement and protects the best one from eviction.

early_stopping has to be stated here. gpt2_medium_1500.py sets it to False -- every arm
of the grid had to reach 1,500 for the sizes to be compared over one window -- and
`import *` carries that in, so the trainer default of True (models/gpt2/train.py,
2026-11-16) would not apply to this file.

The run resumes from a validation loss of 0.5717 at iteration 1,500 and a best of
0.5682 at 1,450, which is the value early stopping has to beat before it counts an
evaluation as no improvement.
"""

from molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_medium_1500_lr6e4 import *  # noqa: F401,F403

max_iters = 3000             # a ceiling; early stopping decides the end
lr_decay_iters = 1500        # unchanged: past it get_lr returns min_lr = 6e-5
init_from = "resume"         # continue the copied checkpoint, do not start over
early_stopping = True
early_stopping_patience = 10
