# What the BERT seed arms measured

Settled 2026-10-29 (put `seed=seed` back) and 2026-11-10 (do not re-run the six).
This document names, for every BERT seed arm, whether the declared seed reached
`transformers`, so that a width quoted from a pair can be called by its right name.
The Japanese version is `BERT_SEED_ARMS_WHAT_THEY_MEASURED.ja.md`.

The point of the document: **three of the widths that have been reported as "the
seed width" are seed widths, and one is not.** They are all about the same size,
0.0012 to 0.0015, which is why the four were easy to read as one thing.

---

## How the seed reached `transformers`, by date

| window | what `TrainingArguments` received | commit |
|---|---|---|
| to 2026-08-02 | no `seed=`; `args.seed` is the library default 42 | -- |
| 2026-08-03 to 2026-10-01 | `seed=seed` and `data_seed=seed` | `05ee336d` |
| 2026-10-02 to 2026-10-29 | `data_seed=seed` only; `args.seed` fell back to 42 | `0f3e7067` |
| since 2026-10-30 | `set_seed(int(seed))` before the model is built, and `data_seed=seed` | `97069327` (PR #340) |

Two measured facts about `transformers` 4.45.1 make the third window matter.

`TrainingArguments.seed` defaults to **42**, so dropping `seed=` does not leave the
seed unset; it silently sets it to the one value the project had standardised on.

`trainer.py` contains **no reference to `data_seed`** -- zero occurrences in the
file. The data order follows `set_seed(self.args.seed)`, which `Trainer.__init__`
calls (`trainer.py:406`) and the training loop calls again (`:2010`). So a config
declaring `data_seed = 1` while `args.seed` is 42 changes nothing at all.

`tests/unit/test_bert_seed_reaches_transformers.py` holds both facts; the second
test fails if a `transformers` upgrade starts reading `data_seed`.

## The six runs whose declared seed did not reach `transformers`

Full scan of the 324 BERT runs, 2026-11-10, `scripts/bert_seed_inventory.py`. Six
rows have `seed_agrees=no`, and they are three pairs:

| run | config asked | `seed.seed` in the manifest | `seed.data_seed` | job | commit |
|---|---|---|---|---|---|
| `bert_medium_24k_lr3e4_seed1` | 1 | 42 | 1 | 199853 | `4a3d9f4` |
| `bert_medium_24k_lr3e4_seed17` | 17 | 42 | 17 | 199855 | `4a3d9f4` |
| `protein_sequence-large-lr5e5-s1` | 1 | 42 | 1 | 152889 | `fd1dbb3` |
| `protein_sequence-large-lr5e5-s17` | 17 | 42 | 17 | 152893 | `fd1dbb3` |
| `protein_sequence-large-lr7e5-s1` | 1 | 42 | 1 | 173558 | `fd1dbb3` |
| `protein_sequence-large-lr7e5-s17` | 17 | 42 | 17 | 173562 | `fd1dbb3` |

All six ran inside the third window (2026-10-06 to 2026-10-22). Every other seed
pair ran before 2026-10-02 and its value is in the manifest as declared. GPT-2 is
not affected: `train.py` calls `torch.manual_seed(seed + ddp_rank)` itself, and the
same scan found 0 disagreements in 343 runs.

## What each reported width measures

`eval_loss_mask`, mean of the last ten evaluations, measured from each run's
`trainer_state.json`.

| pair | `args.seed` per arm | values | width | what varied |
|---|---|---|---|---|
| molnl small 1e-3 | 42 / 1 / 17 | 0.059614 / 0.058306 / 0.058521 | **0.0013** | the seed |
| molnl large 1.7e-4 | 42 / 1 / 17 | 0.057465 / 0.058699 / 0.058008 | **0.0012** | the seed |
| molnl medium 3e-4 | 42 / 42 / 42 | 0.057386 / 0.055869 / 0.056038 | **0.0015** | **nothing that was declared** |
| protein large 5e-5 | 42 / 42 | 2.534990 / 2.540291 | **0.0053** | **nothing that was declared** |
| protein large 7e-5 | 42 / 42 | -- | -- | run stopped at 7,600 and 7,700 of 33,531 steps |

The molnl medium arms are `bert_medium_24k_lr3e4_s42r`, `_seed1` and `_seed17`, all
three launched 2026-10-22 from `4a3d9f4` under the same evaluation definition.

The protein 5e-5 pair's third arm, `protein_sequence-large-lr5e5` at seed 42, is not
comparable with the two: it stopped at 30,000 of 33,531 steps and its best is 2.7526
at step 5,400. The width above is the s1-to-s17 distance alone.

The protein 7e-5 trio was cancelled early, so it yields no width.

**Two arms that declared different seeds and received the same one still differ**,
because the seed was not the only thing left loose. In the third window nothing
seeded the weights before they were built: the model is constructed before
`TrainingArguments` exists, so the first `set_seed` of the process came later, from
inside `Trainer.__init__`. The initial weights therefore came from an unseeded
generator. PR #340 is what moved `set_seed` ahead of the model.

Measured (job 312632, compute node). Two processes that build a small
`BertForMaskedLM` before any seeding report `torch.initial_seed()` of
1479681010697427743 and 14957004836947191761, and weight digests
`e61940822dd90e9c` and `316b7c1ce34f59e0` -- they disagree. Two processes that call
`set_seed(42)` ahead of the model both report seed 42 and digest
`42ff42a10f839006`.

## The sentence a material carries

**The molnl medium width 0.0015 is not a seed width. It is the width between runs
whose declared conditions were identical.** The small 0.0013 and the large 0.0012
are seed widths: those arms received 1 and 17.

A material must not compress the three into "the seed width is 0.0012 to 0.0015".
What the three together do support is narrower and still useful: at this level, runs
that differ in seed and runs that differ in nothing differ by the same amount, so a
gap of this size carries no ranking either way.

## Not re-run

The three pairs are not re-run. 1,500 GPU-h, and a re-run would answer a question
the measurements already answer -- decision of 2026-10-29 §4, confirmed
2026-11-10 §3.3.

## Where the values live

- Per-run seeds, all 324 BERT runs: `report/rt1-bert-seed-inventory-2026-11-10.tsv`,
  columns `config_seed`, `manifest_seed`, `seed_agrees`.
- The run conditions table already carries the mismatch per run, as
  `42 (the config asked for 1. it did not take effect)`:
  `report/run-conditions-20261102.tsv`.
- The widths above: each run's `trainer_state.json`, and the molnl summary in
  `report/rt1-molnl-summary-runs.tsv` (`window_mean`).
- Nothing in `learning_source*` was edited. The manifests record what ran.
