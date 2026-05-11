# Replogle Perturb-seq evaluation

Phase 4 perturbation-response benchmark.  Loads a table of
`(perturbation, baseline, perturbed)` vectors, splits by the
perturbation target, fits a ridge regression on
`(perturbation_embedding -> delta)`, and reports the mean per-
perturbation Spearman / Pearson correlation.
