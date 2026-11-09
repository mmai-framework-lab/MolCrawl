# Adopted checkpoints: genome GPT-2

Settled 2026-10-30. Base series: nine epochs (`gpt2-small-<subset>-ep9`, 21
subsets, gpt2-small, global batch 2,560, `max_iters` per subset).

How to handle these runs downstream: the detector, the comparison against
genome BERT, and the §7 evaluation. The BERT side is
`GENOME_ADOPTED_CHECKPOINTS_BERT.md` (Japanese:
`GENOME_ADOPTED_CHECKPOINTS_BERT.ja.md`). The Japanese version of this document
is `GENOME_ADOPTED_CHECKPOINTS_GPT2.ja.md`.

---

## Why nine epochs are the base

**The step counts line up with BERT.** In all 21 subsets the `max_steps` of
BERT (`bert-small-<subset>-w1026`) equals the `max_iters` of nine-epoch GPT-2
(measured 21 of 21 on 2026-10-30, 122,684 to 163,900). This is the pair with the
compute matched.

**How low a run's loss got is not a reason to pick it.**

## Which checkpoint is adopted

**The saved checkpoint with the lowest val loss. On a tie, the earlier step.**

- nanoGPT rewrites `ckpt.pt` at the top of the run directory only when the val
  loss improves, so that file holds the best-val weights.
- In all 21 subsets both `ckpt.pt` and `checkpoint-<best_step>/` are present on
  disk (verified 2026-10-30).
- Path of the adopted checkpoint:
  `learning_source_genome_runs/gpt2-small-<subset>-ep9/ckpt.pt`
- best@step runs 19,000 to 105,000, median 28,000. Only `mammal_centered` at
  105,000 sits apart.

For `global_random_seed4`, `ckpt.pt` is step 25,000 while the per-run table's
best@step is 23,000. The val loss at the two steps is the same value to the
recorded precision of six decimals, 1.224900: nanoGPT keeps 25,000 and the
per-run table takes 23,000 under the "earlier step on a tie" rule. Neither is
the final step.

## The final model is not used

**The weights at the final step are not adopted.** All 21 runs of the
nine-epoch series end above their best val, by a median of +0.1800 and at most
+0.5958 (`global_random_seed3`). The three-epoch series ends above its best by a
median of +0.0091 and at most +0.1259, so the median here is roughly twenty
times larger. If a use for the final weights comes up, raise it case by case.

`mammal_centered` alone has a gap of +0.0046, and its best@step is late at
105,000. Do not read the shape of the series off that one run.

## Do not line up best val across subsets

**The subsets are different data and carry differences in difficulty. Absolute
values side by side are not a comparison.**

`mammal_centered`'s best val of 1.0875 ranks first of the 21, and part of that
is the subset being easier. **Compare only within one subset.**

## The three-epoch series (`gpt2-small-<subset>`)

**Adopted on 2026-08-21, no longer used.** Under the judgment of 2026-10-30 it
appears neither in the figures and tables nor in the text.

The outputs and checkpoints have not been deleted. `ckpt.pt` at the top of each
run directory holds the best-val weights in 21 of 21 cases (mtime cross-check,
2026-10-30), and the adopted step runs 27,000 to 50,000. Every per-run field is
in `_results/b21-per-run.csv` and the val series itself in
`_results/b21-val-series.csv`.

What was recorded about this series stands, and is not withdrawn:

- **All 21 runs end above their best val.** The reversal is a median of +0.0091
  and at most +0.1259 (`global_random_seed3`); `mammal_centered`'s +0.0760 is
  the third largest. An earlier version of this record said "10 of 21 runs end
  above their best val", which meant 10 runs with a reversal of 0.01 or more;
  it was corrected on 2026-10-30 by counting again from the per-run table.
- 18 of the 21 runs threw a val spike of 0.02 or more at some point, and 8 of
  those recovered fully. Whether a reversal is still there at the end depends
  more on when the spike happened than on how large it was (rank correlations:
  size +0.218, step of onset +0.515, steps remaining -0.427, n=21 -- all
  moderate).
- `mammal_centered` spiked at step 52,000 and the run ended before it
  recovered. Its adopted checkpoint (step 34,000) predates the spike, so it is
  unaffected.

## What is recorded where

The nine-epoch per-run values (best, best@step, final value, reversal, tokens
processed) are the 21 rows whose `series` column is `gpt2-ep9` in
`report/genome-loss-curves/genome-loss-per-run.tsv`, outside the repository. The
val series itself is `genome-loss-curves.tsv` in the same directory (as of
2026-10-20).

## Code that points at a final checkpoint

These three pin a **final step** by step number in the previous generation's run
tree (`learning_source_20260617/genome_sequence/`). None of them feeds any
current figure or table.

- `scripts/run_clinvar_eval_subsets.py`: bert `checkpoint-60000`, gpt2
  `checkpoint-50000`
- `scripts/run_clinvar_finetune_bert_subsets.py`: `checkpoint-60000`
- `scripts/run_clinvar_finetune_gpt2_subsets.py`: `checkpoint-50000`

In the previous generation's 21 runs, 60000 and 50000 are respectively the
largest checkpoint (measured 2026-10-30). A note saying so was put at the top of
each file on 2026-10-30.

The two current paths (`scripts/clinvar_zeroshot_bert_rescore.py` and
`scripts/clinvar_linear_probe.py`) used to fall back to the largest step when
the adopted checkpoint could not be resolved; since 2026-10-30 they **stop**
instead. The step they took is written into the output.
