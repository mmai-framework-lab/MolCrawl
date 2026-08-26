---
name: portable-run-paths
description: How to resolve inputs, outputs and the interpreter in this repo's job scripts so a run behaves the same from the main checkout, a worktree, or another machine. Read before writing or editing anything under workflows/ (sbatch, submitters), before choosing where a run writes checkpoints or results, and before putting any absolute path in tracked source.
---

# Portable paths in run scripts

A job script here can be launched from the main checkout, from a git worktree, or
from a different machine. **Where it reads from, where it writes to, and which
interpreter it uses must not silently change with that.** Every rule below comes
from an incident that reached production.

## The four rules

### 1. No absolute server path in tracked source

Take roots from an environment variable and **fail loudly when it is unset**.
A default that silently points somewhere is worse than an error.

```bash
: "${GENOME_SOURCE_ROOT:?set GENOME_SOURCE_ROOT to the learning_source root holding genome_sequence/}"
[ -d "${GENOME_SOURCE_ROOT}/genome_sequence" ] || {
    echo "GENOME_SOURCE_ROOT=${GENOME_SOURCE_ROOT} has no genome_sequence/ under it" >&2; exit 1; }
```

### 2. Outputs must not be resolved relative to the checkout

**This is the one that costs the most and shows up last.**

`learning_source*` is gitignored, so **every checkout and every worktree has its
own copy of that tree**. A production sweep launched from a worktree writes its
checkpoints into the worktree. Worktrees are meant to be disposable; the
artifacts are not.

*What happened:* the genome BERT sweep ran from a worktree and put 102G of
production checkpoints there, while the GPT-2 results sat in the main checkout.
The permanent record was split across two directories, and the BERT half sat
inside something built to be deleted. Nothing failed — that is why it was only
noticed at aggregation time.

Decide the output root explicitly, the same way inputs are decided:

```bash
RUNS_ROOT="${RUNS_ROOT:-$PWD/learning_source_genome_runs}"   # override to pin it
OUT="${RUNS_ROOT}/bert-small-${SUBSET}"
```

and **say in the log which root was used**, so the answer is in the run's own
record rather than in someone's memory:

```bash
echo "=== outputs: ${OUT} ==="
```

### 3. The interpreter lives in the main checkout, not in a worktree

`./miniconda` does not exist in a worktree. Resolve it through git's common dir
rather than writing the path in:

```bash
if [ -x "./miniconda/bin/torchrun" ]; then
    PYROOT="$PWD"
else
    PYROOT="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)/.." 2>/dev/null && pwd)"
fi
TORCHRUN="${PYROOT}/miniconda/bin/torchrun"
[ -x "${TORCHRUN}" ] || { echo "no torchrun under ${PYROOT}/miniconda -- set PYROOT"; exit 1; }
```

*What happened:* the first production launch died at `rc=127` on
`./miniconda/bin/torchrun: No such file or directory`. It was caught only because
one subset was submitted before the other twenty; all twenty-one would have died
at the same line.

### 4. Do not depend on absolute paths written *into* artifacts

HF stores `best_model_checkpoint` in `trainer_state.json` as an absolute path.
Move the tree and 100+ of those files point at nothing.

Record **step numbers**, not paths, in anything meant to outlive the run —
`_results/` CSVs and `ADOPTION.md` do this. Read the field for its step number
(`int(path.rsplit("-", 1)[1])`) rather than opening the path.

## Before submitting a sweep

- [ ] Every root comes from an env var that fails loudly when unset
- [ ] The output root is stated in the log
- [ ] The script runs from a worktree as well as from the main checkout
- [ ] **Submit one job first.** Rules 1 and 3 were both caught this way, at 0.43
      GPU-h, on work that would otherwise have failed twenty-one times over
- [ ] The submitter has a dry run, and its subset list is derived from something
      on disk (`training_ready_hf_dataset_bert` exists) rather than from a name
      pattern — `gpt2-output` got through a name filter once

## Where things belong

| | |
|---|---|
| Reports for the boss | `/data1/rkp00024/matsubara/report/` — outside any checkout, by agreement |
| Run artifacts, durable record | `learning_source_genome_runs/_results/` in the **main checkout** |
| Scratch, working notes | `tmp/` — **not gitignored**, so never `git add .` |

Run artifacts sitting inside a git working directory is historical accident, not
design. Until a location outside every checkout is agreed, keep them in the main
checkout and never in a worktree.
