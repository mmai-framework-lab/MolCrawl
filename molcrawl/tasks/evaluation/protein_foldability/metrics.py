"""Structure-free foldability proxies.

When ESMFold / AlphaFold2 are unavailable we fall back to distributional
indicators:

* ``mean_length`` / ``std_length`` of generated sequences
* ``amino_acid_kl`` - KL divergence between generated composition and
  the reference corpus composition (natural log)
* ``novelty`` - 1.0 - (fraction of exact matches with the reference)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, Sequence


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _composition(sequences: Iterable[str]) -> Dict[str, float]:
    counter: Counter[str] = Counter()
    total = 0
    for seq in sequences:
        for ch in seq:
            ch = ch.upper()
            if ch in AA_ALPHABET:
                counter[ch] += 1
                total += 1
    if total == 0:
        return {aa: 0.0 for aa in AA_ALPHABET}
    return {aa: counter.get(aa, 0) / total for aa in AA_ALPHABET}


def amino_acid_kl(
    generated: Sequence[str], reference: Sequence[str], epsilon: float = 1e-6
) -> float:
    gen_dist = _composition(generated)
    ref_dist = _composition(reference)
    divergence = 0.0
    for aa in AA_ALPHABET:
        p = gen_dist[aa] + epsilon
        q = ref_dist[aa] + epsilon
        divergence += p * math.log(p / q)
    return divergence


def length_stats(sequences: Sequence[str]) -> Dict[str, float]:
    if not sequences:
        return {"mean_length": 0.0, "std_length": 0.0}
    lengths = [len(s) for s in sequences]
    mean = sum(lengths) / len(lengths)
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    return {"mean_length": float(mean), "std_length": float(math.sqrt(variance))}


def novelty_vs_reference(
    generated: Sequence[str], reference: Sequence[str]
) -> float:
    if not generated:
        return 0.0
    ref_set = set(reference)
    unique_matches = sum(1 for seq in generated if seq in ref_set)
    return 1.0 - unique_matches / len(generated)


def pfam_hit_rate(_generated: Sequence[str]) -> float:
    """Placeholder for HMMER / Pfam hit-rate metric.

    Requires external HMMER installation; returns NaN when the
    dependency is unavailable.  The workflow wiring the actual
    computation is left for a follow-up PR.
    """
    return float("nan")
