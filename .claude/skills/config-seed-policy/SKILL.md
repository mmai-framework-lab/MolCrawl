---
name: config-seed-policy
description: Which seed a new pretrain config gets, and where it is written. Read before creating or copying any config under molcrawl/tasks/pretrain/configs, before adding an arm to an existing grid, before setting up a seed-variance run, and before launching anything whose seed you have not checked.
---

# A new config gets seed 42, and only seed

Directive of 2026-09-30 (igarashi, `all-seed-order-2026-09-30_1.md`), for all five
modalities and both architectures.

```python
seed = 42
```

**`data_seed` is not written.** Neither path reads it, and both now refuse a config
that declares one different from `seed` -- see "One seed, not two" below.

**Existing runs keep the seed they ran with. Nothing is re-run to change a seed.**

## The one exception, and the case that now needs a decision

| Case | Seed | How to recognise it |
|---|---|---|
| A new run, standing on its own | **42** | nothing existing will be tabulated next to it |
| **A run measuring seed variance** | **1 and 17** | the point of the run is the spread itself |
| A point added to a grid whose arms ran at another seed | **ask first** | its number will sit in a table beside arms that already ran |

**On 2026-11-10 every config in the tree was set to 42** (matsubara, carrying the
2026-09-30 directive through the whole tree). The per-config numbers assigned on
2026-08-03, and the seeds the grids ran at -- compounds BERT **9**, protein BERT
**82** and **77**, rna BERT **106**, protein GPT-2 **1001** -- are no longer
anywhere under `configs/`. The 32 variance arms keep 1 and 17.

That removes the number a grid continuation used to copy, and the problem it solved
has not gone away: a rate filled in between two existing rates, a size added to a
ladder, a rerun of one arm with a longer schedule are all read against arms that
already ran, and **a different seed puts seed variance inside a difference that is
supposed to be about the rate or the size.** For a continuation, read the earlier
arms' seed out of their `run_manifest.json`, and then either say in the report that
the new point ran at a different seed, or get a decision before launching. Do not
put the old number back into a config without that decision.

Before writing the number, check what the neighbours use:

```bash
sbatch workflows/check-seed-declarations.sbatch   # every config's seed / data_seed
```

## Where the number goes

**In the config, never on the command line.** A seed passed at launch is invisible
to anyone reading the config afterwards, and the two drift.

`run_manifest.json` records the resolved value, so what a run actually used can be
read back without re-deriving it from the config tree.

## One seed, not two

The directive's §3 asked for `seed` and `data_seed` both. **Neither path can honour
that, so the project writes `seed` alone** (checked 2026-10-02, transformers
4.45.1):

| | |
|---|---|
| BERT | `main.py` has no module-level `data_seed`, so `config_keys` never picks one up. `main.py` passes `data_seed=seed` to `TrainingArguments`, but **`data_seed` appears only in `training_args.py` in this version -- the Trainer never reads it.** `_get_train_sampler` returns a plain `RandomSampler`, so the order comes from the global RNG that `set_seed(seed)` fixed |
| GPT-2 | `train.py` has no `data_seed` at all. One seed drives `torch.manual_seed` (the weights) and `np.random.seed` (`get_batch`'s draw), at `train.py:429-432` |

A declaration that does nothing is worse than no declaration, because it reads as
a separation that was chosen. **Both paths now refuse a config that sets
`data_seed` to something other than `seed`**, with a message saying why. A config
that sets it equal is accepted and ignored.

Separating the two seeds is possible, but it means going around the Trainer to
hand the sampler its own generator, and on the GPT-2 side splitting a seed that
currently drives both draws. Nothing in the five modalities has needed it: the
variance runs (1 and 17) move both together.

## What the seed reaches

| | Affected |
|---|---|
| weight initialisation | yes |
| the order training data is served in | yes, through the same `seed` |
| which tokens BERT masks | yes |
| the sequences GPT-2 draws for evaluation, when it redraws them | yes |
| the train / valid / test split | **no** -- fixed when the data was built |
| the fixed evaluation set's membership | **no** -- fixed under a separate seed |

So a seed change does not make two runs incomparable in their evaluation set; it
changes what the model saw and in what order, which is enough to move a loss by
more than the differences a grid is trying to resolve.

## What "3 seeds" means when a difference is claimed

A ranking between two arms needs the spread behind it, and the spread comes from
the variance runs (1, 17, and the arm's own seed). A difference smaller than the
measured spread is not reported as a ranking -- see the checks in
`run-completion-figures`.
