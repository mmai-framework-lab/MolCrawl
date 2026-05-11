# Protein foldability evaluation

Phase 2 generation-side benchmark.  Samples from a protein decoder and
computes structure-free proxies because the plan explicitly calls out
ESMFold / AlphaFold2 as too heavy for routine runs.

## Metrics

- `mean_length` / `std_length` of generated sequences
- `amino_acid_kl` - KL divergence between generated composition and the
  reference corpus composition
- `novelty` - 1 minus exact-match overlap with the reference corpus
- `pfam_hit_rate` - placeholder (NaN by default).  Follow-up PR will
  wire the HMMER + Pfam-A pipeline when those dependencies are present.

## Entry point

```bash
bash workflows/eval-protein-foldability.sh
```
