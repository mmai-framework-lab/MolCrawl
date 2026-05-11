# ChEMBL scaffold held-out evaluation

Phase 1 internal held-out benchmark.  Splits the training ChEMBL corpus
by Bemis-Murcko scaffold so that the test SMILES cannot be seen during
training.  Two modes:

- **decoder** (`label_column` unset): reports perplexity of the
  held-out SMILES under a GPT-2 compound model.
- **encoder** (`label_column` set + `train_csv` supplied): fits a
  logistic-regression probe on training embeddings and scores the
  held-out split.

## Scaffold splitting

Generation of the CSVs is left for a follow-up PR that wires a
`--scaffold-split` flag into
`molcrawl/compounds/dataset/prepare_chembl.py`.  Once that lands, this
task can consume the emitted `heldout.csv` directly.

## Entry point

```bash
bash workflows/eval-chembl-heldout.sh
```
