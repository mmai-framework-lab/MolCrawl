"""MOSES-family distribution-learning metrics.

Core metrics (validity / uniqueness / novelty / internal_diversity) are
delegated to :data:`molcrawl.tasks.evaluation._base.default_registry`.
This module also adds bootstrap 95 % CIs over the four core metrics,
classifies invalid SMILES into rough failure modes, and reports
generation-vs-reference distribution shape diagnostics (length stats,
heavy-atom KL, element coverage).

Optional FCD / SNN / Fragment / Scaffold metrics from the upstream
``moses`` / ``fcd`` packages are still surfaced via
:func:`optional_extended_metrics` when those dependencies are
installed.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from molcrawl.tasks.evaluation._base import default_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------


def distribution_metrics(
    generated: Sequence[str],
    reference: Sequence[str],
) -> Dict[str, float]:
    """Return the core MOSES metrics."""
    return {
        "validity": float(default_registry.compute("validity", generated)),
        "uniqueness": float(default_registry.compute("uniqueness", generated)),
        "novelty": float(default_registry.compute("novelty", generated, reference)),
        "internal_diversity": float(
            default_registry.compute("internal_diversity", generated)
        ),
    }


def optional_extended_metrics(
    generated: Sequence[str],
    reference: Sequence[str],
) -> Optional[Dict[str, float]]:
    """Best-effort wrapper over the reference ``moses`` package.

    Returns ``None`` when the upstream dependency is unavailable.
    """
    try:
        from moses.metrics import get_all_metrics  # type: ignore
    except ImportError:
        logger.info(
            "Optional dependency 'moses' not available; skipping FCD / SNN / Fragment / Scaffold"
        )
        return None
    try:
        extended = get_all_metrics(
            gen=list(generated), train=list(reference), n_jobs=1, device="cpu"
        )
    except Exception:  # pragma: no cover - upstream is heavy, best-effort
        logger.exception("MOSES reference metric computation failed")
        return None
    out: Dict[str, float] = {}
    for key, value in extended.items():
        try:
            out[f"moses.{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------
# Bootstrap CIs over the four core metrics
# ---------------------------------------------------------------------


def bootstrap_distribution_ci(
    generated: Sequence[str],
    reference: Sequence[str],
    n_boot: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
    skip_internal_diversity: bool = True,
    generated_canonical: Optional[Sequence[Optional[str]]] = None,
    reference_canonical_set: Optional[set] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return ``(lo, hi)`` percentile CIs on the core metrics.

    ``internal_diversity`` is skipped by default because it is O(N²) on
    Morgan fingerprints — running it 100× over 30k molecules dominates
    the wall-clock. Set ``skip_internal_diversity=False`` to include it.

    Performance critical: novelty against MOSES train (≈ 1.6 M SMILES)
    must NOT re-canonicalise the reference per iteration. Callers should
    pass ``generated_canonical`` (one canonical-or-None per generated)
    and ``reference_canonical_set`` (canonical SMILES set) so the loop
    body becomes O(N) per iteration instead of O(N + |reference|) RDKit
    calls. When either is omitted, the function falls back to a one-off
    canonicalisation up front (cheaper than per-iteration but still
    requires RDKit).
    """
    n = len(generated)
    if n < 5 or n_boot <= 0:
        return {}

    if generated_canonical is None or reference_canonical_set is None:
        try:
            from rdkit import Chem, RDLogger

            RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
        except ImportError:
            return {}

        def _canon(s: str) -> Optional[str]:
            if not isinstance(s, str) or not s:
                return None
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return None
            try:
                return Chem.MolToSmiles(mol, canonical=True)
            except Exception:  # noqa: BLE001
                return None

        if generated_canonical is None:
            generated_canonical = [_canon(s) for s in generated]
        if reference_canonical_set is None:
            reference_canonical_set = {c for c in (_canon(s) for s in reference) if c is not None}

    canon_arr = list(generated_canonical)
    ref_set = reference_canonical_set

    # Note: uniqueness is intentionally NOT bootstrapped. Bootstrap uses
    # sampling with replacement, which inflates duplicate counts in the
    # resample even when every molecule in the original pool is unique
    # (every with-replacement resample of size n has ≈ 0.37 n duplicates
    # by construction). The point estimate remains in the main metrics
    # block; the CI here covers validity and novelty only, both of which
    # are per-molecule properties that resample correctly.
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {
        "validity": [],
        "novelty": [],
    }
    if not skip_internal_diversity:
        # Internal diversity still needs RDKit per resample; left to the
        # registry path so callers can opt-in.
        samples["internal_diversity"] = []

    gen_list = list(generated)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_canon = [canon_arr[i] for i in idx]
        valid = [c for c in sub_canon if c is not None]
        samples["validity"].append(len(valid) / n)
        if valid:
            samples["novelty"].append(
                sum(1 for c in valid if c not in ref_set) / len(valid)
            )
        else:
            samples["novelty"].append(0.0)
        if not skip_internal_diversity:
            sub = [gen_list[i] for i in idx]
            samples["internal_diversity"].append(
                float(default_registry.compute("internal_diversity", sub))
            )

    lo_p = 100.0 * alpha / 2.0
    hi_p = 100.0 * (1.0 - alpha / 2.0)
    return {
        k: (float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p)))
        for k, v in samples.items()
        if v
    }


# ---------------------------------------------------------------------
# Failure-mode classification
# ---------------------------------------------------------------------


_BRACKET_RE = re.compile(r"\[([^\]]*)\]")
_RING_RE = re.compile(r"\d")


def classify_invalid(smiles: str) -> str:
    """Return a rough category for why a generated SMILES is invalid.

    The categories are heuristic rather than chemically rigorous; they
    are designed to surface the dominant failure modes to a reviewer
    without invoking RDKit twice.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return "empty"
    s = smiles.strip()
    if "(" in s and s.count("(") != s.count(")"):
        return "paren_mismatch"
    if "[" in s and s.count("[") != s.count("]"):
        return "bracket_mismatch"
    if "." in s:
        return "fragmented"
    rings = _RING_RE.findall(s)
    digit_counts = Counter(rings)
    odd = [d for d, c in digit_counts.items() if c % 2 == 1]
    if odd:
        return "ring_unclosed"
    return "rdkit_parse_failure"


def failure_mode_summary(
    generated: Sequence[str], canonicalised: Sequence[Optional[str]]
) -> Dict[str, int]:
    """Categorise each invalid generated SMILES.

    ``canonicalised[i]`` is the RDKit canonical SMILES (str) when valid,
    ``None`` otherwise — produced upstream once per molecule to avoid
    recomputing here.
    """
    counts: Counter = Counter()
    for raw, canon in zip(generated, canonicalised):
        if canon is not None:
            continue
        counts[classify_invalid(str(raw))] += 1
    return dict(counts)


# ---------------------------------------------------------------------
# Distribution-shape diagnostics
# ---------------------------------------------------------------------


def length_distribution_stats(
    smiles_list: Sequence[str],
) -> Dict[str, float]:
    if not smiles_list:
        return {"n": 0}
    lens = np.array([len(s) for s in smiles_list], dtype=float)
    return {
        "n": int(lens.size),
        "len_mean": float(lens.mean()),
        "len_std": float(lens.std()),
        "len_min": float(lens.min()),
        "len_p25": float(np.percentile(lens, 25)),
        "len_median": float(np.median(lens)),
        "len_p75": float(np.percentile(lens, 75)),
        "len_max": float(lens.max()),
    }


def element_distribution(smiles_list: Sequence[str]) -> Dict[str, float]:
    """Return the relative frequency of common elements across all molecules.

    Counts plain element letters in upper-case (C, N, O, S, F, Cl, Br, I, P)
    on the canonical-SMILES string. Aromatic lower-case `c`, `n`, `o`, `s`
    are folded into their upper-case counterparts. Bracketed atoms
    (e.g. ``[Cu]``) are also counted.
    """
    counts: Counter = Counter()
    total = 0
    for s in smiles_list:
        if not isinstance(s, str):
            continue
        # Bracketed atoms first
        for m in _BRACKET_RE.finditer(s):
            inside = m.group(1)
            sym = "".join(ch for ch in inside if ch.isalpha())
            if sym:
                counts[sym.capitalize()] += 1
                total += 1
        # Plain non-bracketed atoms
        clean = _BRACKET_RE.sub("", s)
        i = 0
        while i < len(clean):
            ch = clean[i]
            if ch in "CNOSPFI":
                counts[ch] += 1
                total += 1
            elif ch in "cnops":
                counts[ch.upper()] += 1
                total += 1
            elif ch == "B" and i + 1 < len(clean) and clean[i + 1] == "r":
                counts["Br"] += 1
                total += 1
                i += 1
            elif ch == "C" and i + 1 < len(clean) and clean[i + 1] == "l":
                counts["Cl"] += 1
                total += 1
                i += 1
            i += 1

    if total == 0:
        return {}
    return {k: float(v / total) for k, v in sorted(counts.items())}


def element_distribution_kl(
    gen_dist: Dict[str, float], ref_dist: Dict[str, float]
) -> float:
    """Return KL(generated || reference) over the element distribution."""
    if not gen_dist or not ref_dist:
        return float("nan")
    keys = set(gen_dist) | set(ref_dist)
    eps = 1e-9
    kl = 0.0
    for k in keys:
        p = gen_dist.get(k, 0.0) + eps
        q = ref_dist.get(k, 0.0) + eps
        kl += p * np.log(p / q)
    return float(kl)
