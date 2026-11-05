# Column convention for the report TSVs

## 0. Status

**Convention version 1.** All five open questions were settled by matsubara on
2026-11-05.

| # | Decision |
|---|---|
| 1 | Adopt a convention. **Existing files are never rewritten in place** |
| 2 | The value column is **named (form B)**. `value`+`metric` (form A) is the exception for a table whose metric differs from row to row. A bare `value` is forbidden. §9 |
| 3 | Column names are ASCII only. Japanese names live in a table on the viewer side |
| 4 | `#` comment lines are allowed. **A provenance block is required** |
| 5 | A non-conforming file is not repaired; **a separate conforming file is written beside it** |

Every count below is output of `model-web/tools/check-tsv.mjs` and can be
re-measured rather than believed.

```
REPORT_DIR=<report directory> node model-web/tools/check-tsv.mjs
```

## 1. Why this is needed

The TSVs under `report/` were each written by a different script, and nothing
declared what a column meant. A reader had to infer it from the name and the
contents, and the following happened.

| What happened | How much |
|---|---|
| `seed` offered as the y axis of a loss chart | 36 files. `job_id` in 13, `job` in 7 |
| A `#` comment line read as the header row, making the whole file unusable | 14 files |
| No header row at all, so the first data row was read as the column names | 9 files |
| A `value` column with nothing in the file saying what the value is | 39 files |
| Two architectures in one file, with the path saying neither | 43 files |

## 2. The file name states the convention

**A conforming file's name begins with `rt<version>-`.**

```
rt1-compounds-loss-curves.tsv
```

`rt` is report-tsv and `1` is the version of the convention. A later version
is `rt2-`, so **the version is known from the name alone** -- which is the
point, because the listing has to decide without reading 56 MB.

**By default the viewer ingests only files marked `rt<version>-`.** It falls
back to showing everything only when there is no marked file at all (better
than an empty page with no stated reason). Unmarked files are still reachable
through the convention filter.

## 3. The provenance block

**The first line declares the version; `# key: value` lines follow with the
provenance.** The version is declared in both the name and the contents, and
the checker and the viewer report a disagreement between them.

```
#! report-tsv 1
# model: rna bert large
# run: rna-bert-large-lr1e4-s42
# checkpoint: checkpoint-15000 (adopted)
# source: learning_source_rna_runs/rna-bert-large-lr1e4-s42/checkpoint-15000/trainer_state.json
# job: 267687
# generated: 2026-11-05T15:57:56+09:00 by workflows/xxx.py
# note: free text, any number of lines
```

| Key | Required | How to write it |
|---|---|---|
| `model` | **required** | `<modality> <arch> <size>`. Several in one file: `several (columns: arch,size,lr)`. A table that belongs to no model: `not applicable (<what the table is>)` |
| `source` | **required** | Where the numbers came from: a run's output, or the TSV this one was derived from |
| `generated` | **required** | `<ISO8601+09:00> by <script>` |
| `checkpoint` | required when there is no progress column | The checkpoint name, or `column: checkpoint` when it differs per run |
| `run` | optional | The run name. Several of them belong in a column |
| `job` | optional | The Slurm job id |
| `note` | optional | Free text. A `#` line that is not in key form is treated as free text too |

A `#` line that is not `# key: value` is prose, and the viewer shows it as it
stands. Some of the comments in the existing files are worth keeping -- for
example `unigram floor for next-token prediction on this corpus: 4.6514`.

## 4. Shape

**Long format: one measured value per row.** The columns say what the value
is, which model it belongs to, and at what point it was taken.

```
arch	size	lr	seed	step	eval_loss_mask
bert	large	1.5e-4	9	100	2.477893
```

**The column name states what the value is** (decision 2, §9). The name
`value` is used only together with a `metric` column, and only when the metric
differs from row to row within the same table.

Wide shapes -- a column per `step`, or a column per run -- are not used. A
table that grows a column for every new series can be neither filtered nor
sorted.

## 5. Column roles

**Every column is one of these seven.** This is the `ROLES` table in
`model-web/lib/tsv.js` itself: the implementation and this document point at
one table, not two.

| Role | Meaning | Examples | Axis |
|---|---|---|---|
| `progress` | How far a run got | `step` `iter` `tokens` `pfdays` | x |
| `measure` | What was measured | `loss` `eval_loss_mask` `auroc` `mrr` `value` | y |
| `param` | Which model this is | `arch` `size` `lr` `seed` `epochs` `panel` | may also be x |
| `measurement` | Which measurement this row is | `split` `metric` `eval_set` | not an axis |
| `id` | Names a thing; not a quantity | `job` `job_id` `checkpoint` | **never an axis** |
| `count` | How many of something | `rows` `batches` `queries` | usable, never a default |
| `other` | A numeric column no rule claims | | usable, never a default |

`seed` is a `param`. **It is part of what makes one model different from
another, so it separates rows, but it is neither an x nor a y axis.** A chart
of loss against a random seed means nothing.

A name the table does not hold is classified by pattern, and `other` when no
pattern fits either.

## 6. The name vocabulary

**Use a name from this table before inventing one.**

| Role | Names |
|---|---|
| Progress | `step` `iter` `tokens` `pfdays` `steps_run` `last_step` `best_step` |
| Value | `loss` `val_loss` `train_loss` `eval_loss_mask` `auroc` `spearman` `accuracy` `recall@1` `mrr` `window_mean` `floor`, and `value` only together with a `metric` column |
| Model | `arch` `size` `lr` `seed` `epochs` `modality` `dataset` `run` |
| Measurement | `split` (`train`/`val`) `metric` `eval_set` |
| Identifier | `job` `job_id` `checkpoint` |
| Count | `rows` `batches` `queries` `pairs` |
| Figure facet | `panel` `series` `group` `member` `subset` |

**A column name holds only ASCII alphanumerics, `_`, `@` and `.`**
(decision 3). The `@` in `recall@1` and `loss@31G` reads as "at" and stays.
Japanese names live in the viewer's `public/glossary.js` (174 of them). A
Japanese column name defeats grep and splits the Japanese wording across two
places.

## 7. The 17 rules the checker applies

A file that breaks a **required** rule either cannot be read by a machine or
will be read wrongly. A **recommended** one can be read, but something is
lost. An **informational** one does not apply to a summary table. R13-R16
apply only to a file that claims the convention in its name.

| | Rule |
|---|---|
| required | R1 A header row exists (the line after any blank and `#` lines) |
| required | R2 Every row has as many fields as the header |
| required | R4 No duplicate column names |
| required | R5 No BOM |
| required | R6 LF line endings only |
| required | R10 A missing value is an empty cell (not `NaN` `NA` `None` `-`) |
| required | R12 A `value` column comes with a `metric` column |
| required | R13 The `rt` version agrees between the name and the contents |
| required | R14 The provenance has `model`, `source` and `generated` |
| recommended | R3 Column names hold only ASCII alphanumerics, `_`, `@`, `.` |
| recommended | R9 There is a numeric column that can be an axis |
| recommended | R11 A file holding several architectures has an `arch` column |
| recommended | R15 The provenance claims nothing is `unknown` (or `不明`, which is what the files written before the convention say) |
| recommended | R16 With no progress column, the provenance has `checkpoint` |
| recommended | R17 A `value`+`metric` pair whose `metric` holds one value becomes a named value column |
| informational | R7 There is a progress column |
| informational | R8 There is a value column |

On the BOM: the reports (`.md`) are written with one so that Slack does not
treat them as binary, but **a TSV carries no BOM**. It ends up inside the
first column's name.

## 8. Where this stands

As of 2026-11-05 18:40 there are 503 TSVs under `report/`: 291 originals plus
212 conforming files. Jobs that are still running write into this tree, so the
counts move.

| | Files |
|---|---|
| Claim the convention (`rt1-`) | 212 |
| ... and keep all 9 required rules | 212 |
| ... and break only R16 (the checkpoint cannot be determined) | 57 |
| Originals with no conforming file yet | 79 |

Whether a conforming file holds the same numbers as its original is checked by
`tools/verify-rt1.mjs`. It follows the provenance's `source` back to the
original, works out the renamed column and the one dropped column from the two
headers, and compares every data row. It does not use make-rt1's code.

```
REPORT_DIR=<report directory> node model-web/tools/verify-rt1.mjs
-> 212 of 212 rt files match their source (584,135 data rows compared)
```

**A conforming file goes stale when its original is rewritten afterwards.**
This has already happened once: the molnl smoltest retrieval ranks grew from
1,000 rows to 2,000. That is the price of keeping both files, and detecting it
is left to this comparison. Until the scripts that regenerate the originals
are changed, run it once before trusting a conforming file.

**The 79 files without one are the ones with a question left that a machine
cannot answer.** The reasons, counted with overlap, are below; one line from a
person is enough for `tools/make-rt1.mjs` to write the file.

| The question left | Files | How to answer it |
|---|---|---|
| Which model is this table about (no model column) | 51 | `--model "<modality> <arch> <size>"`, or `--model "not applicable (<what the table is>)"` |
| What is `value` a value of | 39 | `--metric <name>` (it becomes the value column's name) |
| There is no header row; what are the columns called | 10 | Decide the header, then write the file again |
| Some rows do not have as many fields as the header | 4 | Repair the original, or write the file again |
| What should a non-ASCII column name become | 17 | `--rename <old>=<new>`, repeated per column (the old names are Japanese, which is the problem) |

## 9. The shape of the value column (decision 2)

**A named value column (form B) is the normal form.** `value` + `metric`
(form A) is allowed **only when the metric differs from row to row** within
one table. **A bare `value` is forbidden** (R12).

```
B (normal)     size  lr  step  val_loss  train_loss
A (exception)  arch  size  lr  step  metric      value
```

Why B is the normal form.

1. **B is already the majority.** Of 193 curve files, 122 are B (73 with
   several named columns, 49 with one) and 53 are A (14 `value`+`metric`, 39
   bare `value`). Among the 111 files touched since 2026-10-20 it is 63 B
   against 34 A, so the migration is light.
2. **The column name itself says what the value is.** The y axis then reads
   the file's own column name, with the viewer's Japanese gloss beside it.
   Under A the y axis has one entry, `value`, and what it holds is unknown
   until the reader has also narrowed `metric`.
3. **The 39 bare-`value` files are unreadable as they stand.** Under B that
   state cannot arise.
4. **Train and val at the same step are the same evaluation event**, so they
   belong on one row.

When A is allowed: **a table whose metric differs from row to row** -- a
downstream-evaluation inventory mixing runs that have `auroc` with runs that
have `spearman`, say. B would turn that into a wide table of mostly empty
cells. Of the 212 conforming files, 12 are form A, and every one of them opens
into 50-80% empty cells, because their metric follows the architecture (BERT
`eval_loss_mask`, GPT-2 `val_loss`). Those 12 stay as they are.

**A form A whose `metric` holds a single value becomes B** (R17). The metric
is not varying, so the exception is being used where it does not apply, and
the name of the value stays outside the column name. Two files qualified and
their conforming versions were rebuilt as B (`value` to `val_loss` and to
`loss`, with the `metric` column dropped). The originals were not touched, so
R17 still reports those two.

One control was added on the viewer side. **In a form A file, each value of
`metric` is offered as a y axis of its own** (internally
`value::metric=eval_loss_mask`). Choosing one plots `value` narrowed to that
metric. While the y axis holds a metric, that column appears neither in the
filters nor in the grouping select -- two controls naming one column can
disagree, and the request carries one value for it.

## 10. Migration

1. **A script that writes a new TSV writes the `rt1-` name and the provenance
   block from the start.** `tools/make-rt1.mjs` is a bridge for the files that
   already exist, not a permanent path.
2. **Existing files are not deleted.** The conforming file sits beside the
   original, and its `source` points back at it.
3. When an original is regenerated, change it to emit only the conforming
   file.
4. While both are present, **read the conforming one**. The original stays as
   a record.

model-web works with or without the convention. What it infers today -- the
name vocabulary, the patterns, skipping `#` lines -- stays as backward
compatibility for files that do not claim the convention.
