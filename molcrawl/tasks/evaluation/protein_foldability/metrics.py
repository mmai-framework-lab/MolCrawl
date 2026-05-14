"""Structure-free foldability proxies.

When ESMFold / AlphaFold2 are unavailable we fall back to distributional
indicators:

- ``mean_length`` / ``std_length`` of generated sequences
- ``amino_acid_kl`` — KL divergence between generated composition and
  the reference corpus composition (natural log)
- ``novelty`` — fraction of generated sequences not present in the
  reference corpus
- ``pfam_hit_rate`` — placeholder for HMMER/Pfam scan (NaN when the
  binary is not installed)
- ``bootstrap_distribution_ci`` — 95 % bootstrap CIs on novelty and
  amino_acid_kl, with reference-side pre-computation so the resampling
  loop does not re-scan ≈ 1 M reference sequences per iteration.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------
# Composition / distribution
# ---------------------------------------------------------------------


def _composition(sequences: Iterable[str]) -> Dict[str, float]:
    counter: Counter = Counter()
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
    return _kl_from_dists(gen_dist, ref_dist, epsilon)


def _kl_from_dists(
    gen_dist: Dict[str, float],
    ref_dist: Dict[str, float],
    epsilon: float = 1e-6,
) -> float:
    divergence = 0.0
    for aa in AA_ALPHABET:
        p = gen_dist.get(aa, 0.0) + epsilon
        q = ref_dist.get(aa, 0.0) + epsilon
        divergence += p * math.log(p / q)
    return float(divergence)


def length_stats(sequences: Sequence[str]) -> Dict[str, float]:
    if not sequences:
        return {
            "n": 0,
            "mean_length": 0.0,
            "std_length": 0.0,
            "min_length": 0.0,
            "max_length": 0.0,
            "median_length": 0.0,
        }
    lengths = np.array([len(s) for s in sequences], dtype=float)
    mean = float(lengths.mean())
    variance = float(((lengths - mean) ** 2).mean())
    return {
        "n": int(lengths.size),
        "mean_length": mean,
        "std_length": float(math.sqrt(variance)),
        "min_length": float(lengths.min()),
        "max_length": float(lengths.max()),
        "median_length": float(np.median(lengths)),
        "p25_length": float(np.percentile(lengths, 25)),
        "p75_length": float(np.percentile(lengths, 75)),
    }


def novelty_vs_reference(
    generated: Sequence[str], reference: Sequence[str]
) -> float:
    if not generated:
        return 0.0
    ref_set = set(reference)
    unique_matches = sum(1 for seq in generated if seq in ref_set)
    return 1.0 - unique_matches / len(generated)


def novelty_against_set(
    generated: Sequence[str], reference_set: Set[str]
) -> float:
    if not generated:
        return 0.0
    matches = sum(1 for s in generated if s in reference_set)
    return 1.0 - matches / len(generated)


def pfam_hit_rate(_generated: Sequence[str]) -> float:
    """Placeholder for HMMER / Pfam hit-rate metric.

    Requires external HMMER installation; returns NaN when the
    dependency is unavailable.  The workflow wiring the actual
    computation is left for a follow-up PR.
    """
    return float("nan")


# ---------------------------------------------------------------------
# Per-sequence diagnostics for the predictions log
# ---------------------------------------------------------------------


def per_sequence_counters(sequences: Sequence[str]) -> List[Counter]:
    """Return the per-sequence amino-acid Counter (cached for bootstrap)."""
    out: List[Counter] = []
    for s in sequences:
        c: Counter = Counter()
        for ch in s:
            ch = ch.upper()
            if ch in AA_ALPHABET:
                c[ch] += 1
        out.append(c)
    return out


# ---------------------------------------------------------------------
# Bootstrap CI on novelty + amino_acid_kl
# ---------------------------------------------------------------------


def bootstrap_distribution_ci(
    generated: Sequence[str],
    reference_set: Set[str],
    reference_aa_dist: Dict[str, float],
    n_boot: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Return percentile ``(lo, hi)`` intervals over ``n_boot`` resamples.

    Both reference inputs are pre-computed (set membership for novelty,
    AA distribution for KL) so the loop body is O(N) per iteration even
    when the reference corpus has ≈ 1 M sequences.

    Returns CIs for ``novelty`` and ``amino_acid_kl``. ``length`` stats
    are not bootstrapped (they're already distributional and cheap).
    """
    n = len(generated)
    if n < 5 or n_boot <= 0:
        return {}

    is_novel = np.array([1 if s not in reference_set else 0 for s in generated], dtype=int)
    per_seq = per_sequence_counters(generated)

    rng = np.random.default_rng(seed)
    novelty_samples: List[float] = []
    kl_samples: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        novelty_samples.append(float(is_novel[idx].mean()))

        agg: Counter = Counter()
        for i in idx:
            agg.update(per_seq[i])
        total = sum(agg.values())
        if total <= 0:
            continue
        gen_dist = {aa: agg.get(aa, 0) / total for aa in AA_ALPHABET}
        kl_samples.append(_kl_from_dists(gen_dist, reference_aa_dist))

    lo_p = 100.0 * alpha / 2.0
    hi_p = 100.0 * (1.0 - alpha / 2.0)
    out: Dict[str, Tuple[float, float]] = {}
    if novelty_samples:
        out["novelty"] = (
            float(np.percentile(novelty_samples, lo_p)),
            float(np.percentile(novelty_samples, hi_p)),
        )
    if kl_samples:
        out["amino_acid_kl"] = (
            float(np.percentile(kl_samples, lo_p)),
            float(np.percentile(kl_samples, hi_p)),
        )
    return out


# ---------------------------------------------------------------------
# Reference-side pre-computation
# ---------------------------------------------------------------------


def precompute_reference(
    reference: Sequence[str],
    max_ref_for_set: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Set[str], Dict[str, float]]:
    """Return ``(reference_set, reference_aa_dist)`` ready for bootstrap.

    For novelty we always need the full reference set (membership lookup
    is O(1) regardless of reference size). For amino_acid_kl the full
    composition is informative; we only subsample if ``max_ref_for_set``
    is set, in which case the AA distribution is computed from the
    subsample to keep the smoke pipeline fast.
    """
    if max_ref_for_set is not None and len(reference) > max_ref_for_set:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(reference), size=int(max_ref_for_set), replace=False)
        ref_subset = [reference[int(i)] for i in idx]
    else:
        ref_subset = list(reference)
    return set(reference), _composition(ref_subset)
