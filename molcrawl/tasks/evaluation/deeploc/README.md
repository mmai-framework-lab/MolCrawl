# DeepLoc 2.0 evaluation

Phase 2 subcellular-localisation benchmark.  10-class encoder-probe
evaluation using the DeepLoc 2.0 dataset; follows the cluster-aware
split shipped with the upstream release when `cluster_id` is present.

## Protocol

1. Load `sequence` + `localisation`; map the 10 canonical class names
   to integer labels.
2. Cluster split (or stratified random when clusters are missing).
3. Embed via `ModelAdapter.embed`, fit a logistic-regression probe on
   the train split, evaluate on the test split.
4. Report accuracy, macro F1, and Matthews correlation coefficient.

## Entry point

```bash
bash workflows/eval-deeploc.sh
```
