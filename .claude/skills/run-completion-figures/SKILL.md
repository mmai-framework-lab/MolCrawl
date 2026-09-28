---
name: run-completion-figures
description: What to produce whenever a training run, an arm of a grid, or a whole grid finishes - the loss figure, the per-run numbers read off it, and the checks a figure must pass before it is shown to anyone. Read before reporting that a run completed, before drawing any loss curve, and before writing a slide or a report that quotes a loss value.
---

# A finished run is not reported without its figure

Reporting "the run completed" with a number and no figure has twice led to a
meeting where the result could not be defended. **Every time a run, an arm, or a
grid reaches its last step, draw the loss and read it.** The figure is part of
the result, not decoration added later for a deck.

## The four steps, every time

### 1. Summarise each run as one row

Curves are raw data. Reduce first, then decide what the figure has to show.

```bash
sbatch --export=ALL,GRID_CONFIG=<config.json>,GRID_TSV=<out.tsv> \
       workflows/bert-grid-analysis.sbatch
```

`scripts/bert_grid_summary.py` writes one row per run: held or collapsed, where
it turned, the best loss and the step it happened at, the loss at fixed token
budgets, and the parameter count summed from the checkpoint's safetensors header.
The vocabulary differs per modality, so "small" is not one parameter count and
must not be quoted as one.

### 2. Draw it

- **Curves** (`scripts/plot_eval_loss_series.py`, driven by a JSON config):
  `panels_hf` for HF runs, `panels_nanogpt` for nanoGPT, `lines_hf` for a handful.
- **Relationships** (`scripts/plot_grid_analysis.py`): the rate that holds against
  model size, loss against tokens for each size at its best rate, and where the
  collapses happened. **These belong in the body of a report; the curves belong in
  an appendix.**

Both run on a compute node. Nothing here runs on the login node.

### 3. Read the numbers off it, in writing

For each run state: the last step reached out of the planned one, the best loss
and where it occurred, whether it ended at the model-free floor, and -- when it
collapsed -- how far down it had got before turning. A run that is still going is
said to be still going, with the fraction of the schedule it has covered.

### 4. Run the checks below before showing it

A figure that fails one of these is redrawn, or the deviation is stated out loud
before anyone else sees it.

## The checks

| # | Check | Why it is here |
|---|---|---|
| 1 | x axis is processed tokens (step x tokens/step) or FLOPs, logarithmic, and panels that invite comparison share one range | a step is not the same work at two sizes: large does ~3x small's compute per step |
| 2 | The body shows relationships; individual curves go to an appendix | a wall of curves is evidence of running, not a result |
| 3 | The title states what the data shows -- read the numbers and confirm before writing it | a figure once shipped claiming the opposite of its own data |
| 4 | Best and final values are not mixed in one table, and the evaluation set and scored tokens are the same across the compared runs | MLM scores 20% of tokens; an autoregressive loss scores all of them |
| 5 | No ordering is claimed without seed variance behind it (3 seeds minimum), and the spread is stated | differences of 0.0001 have been quoted against a seed spread of 0.0006-0.0011 |
| 6 | The five conditions are in the caption: step / epoch / training dataset / global batch / tokens seen. Units are nats/token, rows are sequences | |
| 7 | The model-free floor is drawn, per modality | a loss has no meaning without the value a model that learned nothing reaches |
| 8 | Runs still going, runs cut short, and bounds that were never crossed are marked as such (an arrow, not a point) | |
| 9 | Counts are recomputed from the table, never carried over from the previous report | "15 collapsed" survived three revisions after it had become 25 |
| 10 | BERT and GPT-2 losses are never drawn on the same axes | different objectives, not comparable |

## Where things go

- Figures and their TSVs: the report directory outside the checkout, next to the
  deck that uses them. Never inside the repository.
- Configs naming run paths: also outside the checkout. This repository is public.
- The deck: Marp markdown, rendered with `workflows/marp-to-pptx.sbatch`.

## What this looks like in a report

> rna small 3e-4 は 120,960 step に到達し、eval_loss_mask は 2.6960（最良 2.6928 @
> 119,200）。何も学ばない場合の 9.372 から離れたまま下降が続いていた。図: ...

Not:

> rna small が完走しました。loss は 2.70 でした。
