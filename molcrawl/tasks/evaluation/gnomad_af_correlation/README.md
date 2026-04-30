# gnomAD allele-frequency correlation

Phase 3 generation-side metric for genome decoders.  Scores variants by
``variant_ll - reference_ll`` and correlates against the population
allele frequency from gnomAD.  Higher positive Spearman means common
variants score higher than rare ones under the model.
