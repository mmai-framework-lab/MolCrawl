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

## The approved figure: one panel per rate, sizes overlaid

Signed off on 2026-09-28. **A figure that compares model sizes is cut this way**,
and a deviation from it is announced before the figure is drawn.

| Element | What it is |
|---|---|
| Panels | one per learning rate, the sizes drawn on top of each other. A last panel holds each size at *its own* best rate, and says so in the panel title |
| Colour | the model size, meaning the same thing in every panel. The legend carries the non-embedding parameter count: `small 85.7M`, `medium 303.4M`, `large 575.2M` |
| x axis | 学習に使った計算量（PF-days、C = 6ND）, logarithmic, the same range in every panel |
| y axis | the loss, logarithmic when it spans more than a factor of two |
| Floor | the model-free value as a dashed line with its number, in every panel |
| Title | what the reader should take away, plus the axis in parentheses |

The old cut -- a panel per size with the rates coloured inside it -- still belongs
in an appendix, for reading one size's rate sweep. It is not the figure a size
comparison is made from, because the eye cannot carry a colour across panels.

The caption carries, in this order: max_steps and epochs, the training set in
tokens and sequences, the global batch, the tokens seen at completion, the
definition of the unit (1 PF-day = 10^15 FLOP/s x 86,400 s = 8.64e19), where N
came from (counted from the checkpoints, non-embedding), the 6ND caveat and what
it omits, the compute ratio between sizes at equal tokens (3.54x and 6.71x here),
what colour means, whether seed variance was measured, the floor, and which arms
are still running.

## Which x axis, and what 6ND does not include

Three axes answer three different questions. Pick by the claim, not by habit.

| Axis | What it compares | Use it for |
|---|---|---|
| processed tokens | data efficiency -- what was learned from the same amount of data | the default. Learning rates within one size, data-value arguments |
| compute, C = 6ND, **printed in PF-days** | compute efficiency -- how far the loss falls for the same work | claims that cross model sizes or architectures, scaling laws |
| GPU-hours | money and wall clock, including how well the hardware was used | budget and scheduling appendices |

Tokens and FLOPs are the same axis up to a constant **within one model size**, and
are not the same across sizes: at equal tokens, medium costs 3.54x and large 6.72x
what small costs here (85.6M / 303M / 575M non-embedding parameters).

The unit on that axis is the **petaflop/s-day**: 1 PF-day = 10^15 operations per
second for 86,400 seconds = 8.64 x 10^19 operations. It is a quantity of work, not
a rate and not work per day -- rate (FLOPS, capital S) is a throughput metric and
never an x axis, and GPU-days are hardware time, a third axis. For scale, our
BERT runs are 0.37 to 12.7 PF-days each; GPT-3 was 3,640.

Three things to get right before quoting a FLOPs number:

1. **N is the non-embedding parameter count.** Embeddings barely enter the matrix
   multiplies. molnl's vocabulary of 50,264 puts 39M parameters into embeddings, so
   by total parameters molnl small (125M) looks 1.45x protein small (86M) while
   both cost exactly the same to train.
2. **6ND drops the attention term that grows with sequence length.** At sequence
   length 1,024 it omits 13-18% depending on size. It cancels out in a ratio
   between sizes, but an absolute figure needs the caveat written down.
3. **Equal FLOPs does not make two losses comparable.** MLM scores the masked 20%,
   an autoregressive loss scores every token. A cost axis can be shared; the
   meaning of the y axis cannot. To put BERT and GPT-2 on one figure, the y axis
   has to be a downstream metric, not loss.

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
| 11 | Colour means the same thing in every panel, or the caption says it does not | panels with different rate sets reuse the same four colours |
| 12 | Differences the figure cannot resolve (a log axis at the tail) are given as a table as well | 0.06 and 0.07 sit on top of each other; the ranking is not readable from the curve |

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
