---
name: config-seed-policy
description: Which seed a new pretrain config gets, and where it is written. Read before creating or copying any config under molcrawl/tasks/pretrain/configs, before adding an arm to an existing grid, before setting up a seed-variance run, and before launching anything whose seed you have not checked.
---

# A new config gets seed 42

Directive of 2026-09-30 (igarashi, `all-seed-order-2026-09-30_1.md`), for all five
modalities and both architectures.

```python
seed = 42
data_seed = 42
```

**Existing runs keep the seed they ran with. Nothing is re-run to change a seed.**

## The two exceptions, and how to tell which case you are in

| Case | Seed | How to recognise it |
|---|---|---|
| A new run, standing on its own | **42** | nothing existing will be tabulated next to it |
| **A point added to an existing grid** | **the seed that grid already uses** | its number will sit in a table beside arms that already ran |
| **A run measuring seed variance** | **1 and 17** | the point of the run is the spread itself |

The middle row is the one that gets missed. A rate filled in between two existing
rates, a size added to a ladder, a rerun of one arm with a longer schedule -- all
of these are read against arms that already exist, and **a different seed puts
seed variance inside a difference that is supposed to be about the rate or the
size.** The compounds BERT grid runs at **seed 9**; an arm added to it stays at 9.

Before writing the number, check what the neighbours use:

```bash
sbatch workflows/check-seed-declarations.sbatch   # every config's seed / data_seed
```

## Where the number goes

**In the config, never on the command line.** A seed passed at launch is invisible
to anyone reading the config afterwards, and the two drift.

Both `seed` and `data_seed` are written, even when they are equal: they do
different jobs, and leaving one out means it falls back to a framework default
that no one chose. As of 2026-09-30 **no config declares `data_seed`** -- the BERT
side passes `data_seed=seed` from `bert/main.py:642`, so the manifest records 42
for a config that never said so, and the GPT-2 side has no such knob at all. A new
config writes both so the intent is in the file rather than in the framework.

`run_manifest.json` records the resolved values, so what a run actually used can
be read back without re-deriving it from the config tree.

## What the seed reaches

| | Affected |
|---|---|
| weight initialisation | yes |
| the order training data is served in | yes (`data_seed`) |
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
