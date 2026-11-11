# Protein BERT MLM diagnostic configs, 2026-08 to 2026-09

Six files that drove eleven protein BERT training runs while sitting untracked in the
main checkout. This document is the reason they are in the repository, and the map from
each file to what it produced.

## Why they were untracked, and why they are committed now

They were written during the investigation into protein BERT small collapsing to the
unigram marginal (production run 24870). Each one isolates a single change against
`bert_small.py`, so they were created, launched and edited in place without ever being
committed. The leftovers sweep of 2026-10-30 found them, moved or committed 62 other
untracked files, and deliberately left these six in place pending a decision.

The tree-wide seed change of 2026-11-10 made the omission visible: a checker that scans
the working directory, not git, reports these five configs as the only ones left off
seed 42. Committing them puts the files the runs were launched from under version
control, which is the point of this commit; what to do about their seed is a separate
decision, recorded at the end.

**They are committed as they stand, which is not exactly as they ran.** Every job log
echoes the whole config file once per rank, and comparing those echoes with these files
shows three differences:

| difference | files | as echoed in the log | as committed here |
|---|---|---|---|
| `save_on_improve = True` added after the runs | all five configs | absent | present |
| `per_device_eval_batch_size` | `caseA`, `caseC` | 8 | 160 |
| `gradient_accumulation_steps` came from the launch, not the file | `bert_small_mlm015_diag.py` | `5 * 16`, overridden to 4 at launch | `5 * 16` |

So for the exact text a run used, read the log, not the file. The logs are in
`workflows/slurm-logs/` (git-ignored, on the cluster).

## The files

| file | what it changes against `bert_small.py` | sha256 (committed copy) |
|---|---|---|
| `configs/protein_sequence/bert_small_mlm015_diag.py` | train mask 0.20 -> **0.15**, old-style config (`adam_beta2` left at the GPT-2 spec 0.95) | `22e4bdbc507e92d0…` |
| `configs/protein_sequence/bert_small_esmopt_diag.py` | **`adam_beta2` 0.95 -> 0.999 and `warmup_steps` 200 -> 2000** only ("the ESM/BERT-standard optimizer") | `7271268bb0857ee2…` |
| `configs/protein_sequence/bert_small_esmopt_lr4e4_diag.py` | the same two, plus LR 3e-4 -> **4e-4** | `e19f28f96113d266…` |
| `configs/protein_sequence/bert_small_esmopt_caseA_lr1e4.py` | case A: LR 3e-4 -> **1e-4**, micro-batch 160x4 | `97ee8a33731ab98d…` |
| `configs/protein_sequence/bert_small_esmopt_caseC_mask040.py` | case C: train mask 0.20 -> **0.40**, micro-batch 160x4 | `dc20860876e041cb…` |
| `models/esm2/main_bertswap_diag.py` | the dormant esm2 entry point with `EsmConfig`/`EsmForMaskedLM` swapped for `BertConfig`/`BertForMaskedLM` ("work 1": does the same BERT model learn protein MLM under the esm2 pipeline?) | `f04d2b2a8c0b8f68…` |

## What each file produced

| run | from | job / date | code at run time | result (best) |
|---|---|---|---|---|
| `runs/bert-small-mlm015-eb128` | `mlm015_diag` | 33280, 2026-08-21 | not recorded | `eval_loss` 2.6284 @1,800; cancelled at 14,000 of 20,000 |
| `runs/bert-small-warmup2k-mlm015-eb128` | `mlm015_diag` + `warmup_steps=2000` | 33760, 2026-08-21/22 | not recorded | `eval_loss` 2.5266 @19,500; 20,000 steps, rc=0 |
| `runs/_superseded/esm2script-bertmodel-diag-stopped40664` | `main_bertswap_diag.py` | 40664, 2026-08-25 | not recorded | 2.561 (blended metric, mask 0.15, eff-batch 128); superseded, `SUP-BERT-BLENDED015` |
| `runs/bert-small-esmopt-diag` | `esmopt_diag` | 44835 then resumed by 53852, 2026-08-25/27 | `18bb59c`, then `32b4a01` | `eval_loss` 2.5649 @2,800; 11,177 steps (3 epochs), rc=0 |
| `runs/bert-small-esmopt-lr4e4-diag` | `esmopt_lr4e4_diag` | 53853, 2026-08-28/30 | `32b4a01` | `eval_loss` 2.5630 @2,600; 11,177 steps, rc=0 |
| `runs/bert-caseA-lr1e4` | `caseA_lr1e4` | 77270 (stopped at step 330) then 77562, 2026-09-02/05 | `bcd67bf`, then `dbbc980` | `eval_loss_mask` 2.7264 @11,000; 11,177 steps, rc=0 |
| `runs/bert-caseC-mask040` | `caseC_mask040` | 77271 then 77547, 2026-09-02/05 | `bcd67bf`, then `dbbc980` | `eval_loss_mask` 2.6813 @11,000; 11,177 steps, rc=0 |
| `protein-bert-batchdiag/bert-batchdiag-eb128-lr4e4` | `caseA_lr1e4` + launch overrides | 97111 then 97124, 2026-09-11/12 | `3e0d4e7` | `eval_loss_mask` 2.7585 @9,000; 37,266 steps |
| `protein-bert-batchdiag/bert-batchdiag-eb640-lr4e4` | same | 97112 then 97125 | `3e0d4e7` | `eval_loss_mask` 2.6797 @7,200; 7,453 steps |
| `protein-bert-batchdiag/bert-batchdiag-eb2560-lr4e4` | same | 97113 then 97126 | `3e0d4e7` | `eval_loss_mask` 2.7720 @1,800; 1,863 steps |
| `protein-bert-batchdiag/bert-batchdiag-eb2560-lr3e4` | same | 97114 then 97127 | `3e0d4e7` | `eval_loss_mask` 2.7826 @1,800; 1,863 steps |

`runs/` is `learning_source_20260730_protein_uniref50/protein_sequence/runs/`;
`protein-bert-batchdiag/` is a separate directory outside the repository and outside any
`learning_source` tree.

## As-ran settings, measured from the outputs

Read from each run's newest `checkpoint-*/training_args.bin`, and from
`run_manifest.json` where one exists.

| run | LR | warmup | `adam_beta2` | micro x accum x GPU = eff-batch | train mask | best-model metric | seed |
|---|---|---|---|---|---|---|---|
| `bert-small-mlm015-eb128` | 3e-4 | 200 | 0.95 | 8 x 4 x 4 = 128 | 0.15 | `eval_loss` | 82 |
| `bert-small-warmup2k-mlm015-eb128` | 3e-4 | 2,000 | 0.95 | 8 x 4 x 4 = 128 | 0.15 | `eval_loss` | 82 |
| `bert-small-esmopt-diag` | 3e-4 | 2,000 | 0.999 | 8 x 80 x 4 = 2,560 | 0.20 | `eval_loss` | 82 |
| `bert-small-esmopt-lr4e4-diag` | 4e-4 | 2,000 | 0.999 | 8 x 80 x 4 = 2,560 | 0.20 | `eval_loss` | 82 |
| `bert-caseA-lr1e4` | 1e-4 | 2,000 | 0.999 | 160 x 4 x 4 = 2,560 | 0.20 | `eval_loss_mask` | 82 |
| `bert-caseC-mask040` | 3e-4 | 2,000 | 0.999 | 160 x 4 x 4 = 2,560 | **0.40** | `eval_loss_mask` | 82 |
| `bert-batchdiag-eb128-lr4e4` | 4e-4 | 3,727 | 0.999 | 32 x 1 x 4 = 128 | 0.20 | `eval_loss_mask` | 82 |
| `bert-batchdiag-eb640-lr4e4` | 4e-4 | 745 | 0.999 | 160 x 1 x 4 = 640 | 0.20 | `eval_loss_mask` | 82 |
| `bert-batchdiag-eb2560-lr4e4` | 4e-4 | 186 | 0.999 | 160 x 4 x 4 = 2,560 | 0.20 | `eval_loss_mask` | 82 |
| `bert-batchdiag-eb2560-lr3e4` | 3e-4 | 186 | 0.999 | 160 x 4 x 4 = 2,560 | 0.20 | `eval_loss_mask` | 82 |

Common to all of them: `lr_scheduler_type=linear`, `weight_decay=0.01`,
`max_grad_norm=1.0`, `adam_beta1=0.9`, `adam_epsilon=1e-8`, `seq_len=1024`,
**`bf16=False` and `fp16=False`, i.e. plain fp32**, `eval_steps=100`, `save_steps=2000`,
`DocumentMaskingCollator` (document masking on), data
`learning_source_20260730_protein_uniref50` (train 9,538,464 rows, eval 10,000 rows).
Collapse detection: grace 25,000 for the mlm015 pair, 15,000 for the esmopt pair, and
`degenerate_loss_threshold = None` for case A and case C (detector off for protein,
2026-08-26).

## Where the record of each run lives

| item | config file | job log | `training_args.bin` | `run_manifest.json` | reports |
|---|---|---|---|---|---|
| seed, LR, warmup, `adam_beta2`, batch, `max_steps` | yes | yes | yes | 6 of 11 runs | yes |
| launch overrides | no | yes | yes (as resolved) | yes (`sources.*.from`) | yes |
| **train mask probability** | yes | yes | **no attribute** | **`objective.mlm_probability: null`** | yes |
| code commit | n/a | from 2026-08-25 on | no | yes | yes |
| losses per eval point | n/a | yes | `trainer_state.json` | no | yes |

So the mask probability -- the one thing case C and the mlm015 pair are about -- is in no
run output. It survives in the job log's echo of the config, in these files, and in the
reports under `/data1/rkp00024/matsubara/report/` (27 to 33 files reference each run
name; case C is written up as `case C (mask 0.40) 2.695 vs case A (LR 1e-4) 2.738`).

The manifests of case A, case C and the four batchdiag runs also list the untracked
files as `run.git.dirty_files`, so the fact that a run was launched from an untracked
config is itself recorded in the output.

## Two things to settle, not settled here

1. **Seed.** These five configs declare `seed = 82`, which is what the runs used. Since
   2026-11-10 every config in the tree declares 42, and both
   `scripts/check_seed_declarations.py` and `tests/unit/test_config_seed_42.py` fail on a
   config that is off 42 without being a variance arm. If this branch and the seed branch
   both land as they are, those five files are what fails. Either set them to 42 -- the
   82 stays recorded in the outputs, the logs and this document -- or name them as
   exceptions in the checker.
2. **Relaunching.** None of these files reproduces its run unchanged: the mlm015 pair's
   effective batch came from a launch override, and the three post-run edits listed above
   are in the files. Treat them as the record of an investigation, not as runnable
   production configs.
