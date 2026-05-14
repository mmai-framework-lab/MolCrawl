# ChEMBL scaffold held-out evaluation

Phase 1 internal held-out benchmark. Splits the training ChEMBL corpus
by Bemis-Murcko scaffold so that the test SMILES cannot be seen during
training. Two modes:

- **decoder** (`label_column` unset): reports perplexity of the
  held-out SMILES under a GPT-2 compound model.
- **encoder** (`label_column` set + `train_csv` supplied): fits a
  logistic-regression probe on training embeddings and scores the
  held-out split.

## Building the split

`prepare_csv.py` consumes any one-SMILES-per-line text file (the
training pipeline writes one to
`$LEARNING_SOURCE_DIR/compounds/chembl/chembl_db/smiles.txt`) and emits
a scaffold-disjoint `train.csv` + `heldout.csv` pair:

```bash
python -m molcrawl.tasks.evaluation.chembl_scaffold_heldout.prepare_csv \
    --source-smiles  $LEARNING_SOURCE_DIR/compounds/chembl/chembl_db/smiles.txt \
    --output-dir     $LEARNING_SOURCE_DIR/eval/chembl_scaffold_heldout \
    --heldout-frac   0.05 \
    --max-source     200000 \
    --max-train      50000 \
    --max-heldout    5000 \
    --seed           42
```

Heldout scaffolds are guaranteed disjoint from train (rare scaffolds /
singletons go to heldout, frequent scaffolds to train); the evaluator
also runs a paranoia check via `warn_on_scaffold_overlap` whenever the
encoder probe mode is used.

## Running the evaluator

```bash
MODEL_PATH=.../ckpt.pt \
TOKENIZER_PATH=assets/molecules/vocab.txt \
HELDOUT_CSV=$LEARNING_SOURCE_DIR/eval/chembl_scaffold_heldout/heldout.csv \
OUTPUT_DIR=experiment_data/eval/chembl_scaffold_heldout \
MAX_EXAMPLES=2000 BOOTSTRAP=100 SEED=42 \
bash workflows/eval-chembl-heldout.sh
```

Outputs in `$OUTPUT_DIR`:

- `metrics.json` — perplexity (or AUROC/AUPRC/accuracy/F1) plus
  `bootstrap_ci_95` and SMILES-length stats.
- `predictions.jsonl` — one record per held-out SMILES with
  per-row log-likelihood / perplexity (or probe score + label).
- `predictions.txt` — narrative preview. Perplexity mode samples best-
  and worst-fit SMILES so the reader can see what the model
  recognises vs. what surprises it.
- `REPORT.md` — markdown summary.

## 足固め features

- length-stratified subsample (replaces legacy `df.head(max_examples)`)
- bootstrap 95 % CI on perplexity / probe metrics
- per-row predictions JSONL + best/worst-fit narrative TXT
- scaffold-leak detection in encoder probe mode
