# Adopted checkpoints: genome BERT

Settled 2026-10-30. Base series: nine epochs (`bert-small-<subset>-w1026`, 21
subsets, bert-small, window 1,026, global batch 2,560, LR 1e-4, `max_steps`
122,684 to 163,900 depending on the subset).

How to handle these runs downstream: ClinVar zero-shot, the linear probe, and
comparisons against GPT-2. The GPT-2 side is
`GENOME_ADOPTED_CHECKPOINTS_GPT2.md` (Japanese:
`GENOME_ADOPTED_CHECKPOINTS_GPT2.ja.md`). The Japanese version of this document
is `GENOME_ADOPTED_CHECKPOINTS_BERT.ja.md`.

---

## Why nine epochs and window 1,026 are the base

**The step counts line up with GPT-2.** In all 21 subsets `max_steps` equals
the `max_iters` of nine-epoch GPT-2 (`gpt2-small-<subset>-ep9`), measured 21 of
21 on 2026-10-30. This is the pair with the compute matched.

**Window 1,026 is the shape the model was trained on.** `[CLS]` + 1,024 bases +
`[SEP]`, which puts the same number of bases in the window as GPT-2's 1,024
(`workflows/genome-bert-train.sbatch`).

**How low a run's loss got is not a reason to pick it.**

## Which checkpoint is adopted

**The saved checkpoint with the lowest `eval_loss_mask`. On a tie, the earlier
step.**

Not `eval_loss`. The overall value is diluted by the 10 per cent of positions
where the answer stays in the input, which makes the difference between the
three levels harder to see. Once `eval_loss_mask` was settled on as the
deciding metric, checkpoint selection follows the same metric.

- Path of the adopted checkpoint:
  `learning_source_genome_runs/bert-small-<subset>-w1026/checkpoint-<adopted step>/`
- The adopted step runs 99,000 to 153,000, median 142,000. As a position within
  the run that is 69.2 to 100.0 per cent, median 94.3 per cent; only
  `eukaryote_matched_random_seed1` at 69.2 per cent sits apart.
- Evaluation is every 1,000 steps and saving is every 1,000 steps, so **the
  evaluation points and the save points coincide.** The three-epoch series paid
  a price here -- an evaluation point better than anything saved -- and this
  series does not.
- The adopted checkpoint is present on disk in all 21 subsets (verified
  2026-10-30).
- The checkpoint recorded in the downstream outputs is the adopted one in 21 of
  21 cases (147 combinations, 7 series x 21, cross-checked 2026-10-30).

**`metric_for_best_model` tracks the metric selection uses.** HF protects only
the best of `metric_for_best_model`, so if the two differ the adopted
checkpoint can be deleted by the FIFO. For these 21 runs the two agreed in the
full cross-check of 2026-10-28.

## The final model is not used

**The weights at the final step are not adopted.** The reversal -- final value
minus best -- is 0.0000 to +0.0039, median +0.0015, and 20 of the 21 runs end
above their best (`eukaryote_matched_random_seed6` is the one whose best is the
final evaluation point).

**A small reversal is not evidence that the final value is stable.** No
repeat-run reproducibility has been measured for BERT. On the GPT-2 side "the
best is stable, the final value is not" has been measured (the nine-epoch
series: median +0.1800, maximum +0.5958), and there is no reason to make an
exception on the side that has not been measured.

## Do not line up best val across subsets

**The subsets are different data and carry differences in difficulty. Absolute
values side by side are not a comparison.**

The adopted values run 1.0693 to 1.2752, a spread of 0.2059, and that spread is
not a difference in performance. **Compare only within one subset.**

## Do not compare against the GPT-2 values

GPT-2 is CLM (left context, every position); BERT is MLM (both sides, masked
positions only). Both vocabularies are 10, so the numeric ranges overlap, but
**they are not the same measuring stick.** The validation sets are built
differently too: BERT fixes the first 10,000 rows of valid, GPT-2 draws 64,000
rows with replacement each time. A performance comparison needs the same
downstream task evaluation.

## The ClinVar zero-shot value is taken over the whole window

Judgment of 2026-11-09. Four summing ranges are written (`score_window` over
1,024 positions, `score_flank128` over 257, `score_flank32` over 65,
`score_point` over 1), and **the values in the text and the tables are taken
from `score_window`.**

Three reasons.

It is the range that was decided before the measurement. Moving to a higher one
afterwards would mean the range had been chosen by the result.

GPT-2 can only produce the whole window. It is the one range available in
common to a table covering both architectures.

The conclusion does not change with the range. In all four ranges the highest
run is `global_random_seed4` and the lowest is `mammal_centered`, and the
reordering is confined to one or two adjacent pairs.

The four-range table
(`report/genome-zeroshot-ranges-21-2026-11-02.tsv`) is carried alongside as
evidence that the conclusion does not depend on how the range is taken. It is
not used to replace the adopted value.

**No single AUROC pooled over the three chromosomes is reported.** The
pathogenic rate is 20.6 per cent on chr21, 20.1 on chr22 and 48.2 on chrX, so a
pooled value is not the same quantity as the per-chromosome ones. For the
model-free control the pooled value, 0.5493, falls below every per-chromosome
value it is made of (0.6213 to 0.6534). Controls and runs are both written per
chromosome, and where a summary is needed it is the mean weighted by the
variant counts, with the table saying so. This is the treatment
`genome-probe-judgment-2026-10-27` §2 gave the probe's `overall`.

## The three-epoch, window-512 series (`bert-small-<subset>`, series `bert-base`)

**Adopted on 2026-08-25, no longer used.** The judgment of 2026-10-30 makes
nine epochs at window 1,026 the base.

The outputs and checkpoints have not been deleted. The adopted step runs 82,000
to 109,000; every per-run field is in `_results/bert21-per-run.csv` and the
evaluation series itself in `_results/bert21-eval-series.csv` (21,914 points).
All four metrics (`eval_loss`, `eval_loss_mask`, `eval_loss_copy`,
`eval_loss_random`) are kept in both files.

What was recorded about this series stands, and is not withdrawn:

- No run failed to yield an adopted checkpoint (21 of 21 took the saved
  checkpoint with the lowest `eval_loss_mask`).
- **`metric_for_best_model` was `eval_loss`**, so HF's mark and the adoption
  disagree in 9 runs. The adopted checkpoints survived because HF was
  protecting 10 of them and **11 happened to fall inside the most recent
  `save_total_limit=5`**. A run whose best came in mid-training would have lost
  it.
- Evaluation was every 100 steps and saving every 1,000, so the adopted
  checkpoint differs from the true best over all evaluation points by +0.0014
  on average, +0.0029 at most. The two coincided in only 2 of 21 runs.
- The reversal is +0.0012 to +0.0057, median +0.0026, and all 21 runs end above
  their best.
- **`best_model_checkpoint` in `trainer_state.json` is an absolute path, and
  105 files point at a path that does not exist** (the move from a worktree to
  the main checkout on 2026-08-26). Resume, this document and the CSVs are
  unaffected, because they record the step number. Do not read that field
  directly: take the adopted step from the `adopted_step` column of
  `bert21-per-run.csv`.

## What is recorded where

The nine-epoch per-run values (best, best@step, final value, reversal, tokens
processed) are the 21 rows whose `series` column is `bert-w1026` in
`report/genome-loss-curves/genome-loss-per-run.tsv`, outside the repository. The
evaluation series itself is `genome-loss-curves.tsv` in the same directory (as
of 2026-10-20).

## Code that points at a final checkpoint

These three pin a **final step** by step number in the previous generation's run
tree (`learning_source_20260617/genome_sequence/`). None of them feeds any
current figure or table; a note saying so was put at the top of each file on
2026-10-30.

- `scripts/run_clinvar_eval_subsets.py`: bert `checkpoint-60000`, gpt2
  `checkpoint-50000`
- `scripts/run_clinvar_finetune_bert_subsets.py`: `checkpoint-60000`
- `scripts/run_clinvar_finetune_gpt2_subsets.py`: `checkpoint-50000`

The two current paths (`scripts/clinvar_zeroshot_bert_rescore.py` and
`scripts/clinvar_linear_probe.py`) used to fall back to the largest step when
the adopted checkpoint could not be resolved; since 2026-10-30 they **stop**
instead. The step they took is written into the output.
