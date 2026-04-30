# TAPE evaluation

Phase 2 encoder benchmark covering the five TAPE tasks: secondary
structure (SS3/SS8), contact prediction, remote homology, fluorescence,
and stability.

## Protocol

Each sub-task loads the three JSON splits distributed with TAPE, embeds
the train split through `ModelAdapter.embed`, fits a linear probe
(logistic regression for classification, ridge for regression), and
reports the metric pack in `metrics.py`.  Contact prediction is wired
as a placeholder because the upstream protocol needs residue-residue
logits + PDB-specific masking; the evaluator emits NaN for that task
until the full implementation lands.

## Entry point

```bash
bash workflows/eval-tape.sh
```
