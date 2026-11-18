"""ProteinGym substitution scoring by masked marginals (Issue #388, M2 / step 6).

The official ProteinGym protocol scores a BERT-style masked language model on
substitution assays with *masked marginals*: for each variant,

    score = sum_i  logP(mut_i | reference, position i masked)
                 - logP(wt_i  | reference, position i masked)

summed over the substituted positions ``i``. The masked log-probability rows
depend only on the reference (wildtype) sequence, so they are computed once per
reference and reused across every variant that shares it. Total cost is
therefore ~ O(seq_len) forward passes per assay, *independent of the number of
variants* -- in contrast to the whole-sequence pseudo-log-likelihood delta
(:meth:`HfMlmAdapter.score_likelihood`), which re-masks the full sequence for
every variant (cost ~ O(n_variants x seq_len)) and does not scale to the large
assays (e.g. HIS7_YEAST, ~500k variants).

Substituted positions are found by diffing the reference and mutant *token-id*
streams rather than by parsing the ``A24G`` notation, so the scorer needs no
assumptions about residue<->token indexing beyond "a substitution preserves the
tokenised length". A mutant whose tokenised length differs from the reference
(an insertion/deletion, not a pure substitution) is scored ``NaN``; indels use
the whole-sequence PLL path instead.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def score_substitutions_masked_marginals(
    adapter: Any,
    reference: str,
    mutated_sequences: Sequence[str],
    context_length: int = 512,
) -> np.ndarray:
    """Score substitution variants against one reference by masked marginals.

    ``adapter`` must expose :meth:`masked_marginal_logprobs` (the HF MLM
    adapter); GPT-2 is causal and has no masked marginal. Returns a float
    array aligned to ``mutated_sequences``. A variant identical to the
    reference scores exactly ``0.0`` (no differing positions); a variant whose
    tokenised length differs from the reference scores ``NaN``.
    """
    ref_ids, rows = adapter.masked_marginal_logprobs(
        reference, context_length=context_length
    )
    n_ref = len(ref_ids)
    scores = np.full(len(mutated_sequences), np.nan, dtype=float)
    if n_ref == 0:
        return scores

    for k, mut in enumerate(mutated_sequences):
        mut_ids = adapter._encode(str(mut), max_length=context_length)
        if len(mut_ids) != n_ref:
            # Length change -> insertion/deletion, not a pure substitution.
            continue
        total = 0.0
        ok = True
        for i in range(n_ref):
            if mut_ids[i] == ref_ids[i]:
                continue
            row = rows.get(i)
            if row is None:
                # A differing position that was not scored (a special token):
                # the diff is outside the maskable body -> fall back to NaN.
                ok = False
                break
            total += float(row[mut_ids[i]] - row[ref_ids[i]])
        scores[k] = total if ok else np.nan
    return scores
