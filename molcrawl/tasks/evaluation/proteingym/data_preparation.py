"""ProteinGym table loader.

Supports two on-disk schemas:

1. Legacy (pre-2025 Zenodo release): ``mutated_sequence`` and
   ``wildtype_sequence`` both present as full sequences.
2. Current Zenodo release (record 15293562 and later): each per-assay
   CSV only ships ``mutant`` (e.g. ``"M1I"`` or ``"M1I:K12L"``) and
   ``mutated_sequence``. The wildtype is reconstructed here by reverting
   the listed substitutions on any one mutated row.

In either case the loader guarantees the returned dataframe has the
three columns downstream evaluators expect:

* ``mutated_sequence``  - variant sequence as a string
* ``wildtype_sequence`` - reference sequence (shared across all rows
                          from a given ProteinGym CSV)
* ``DMS_score``         - experimental fitness label (float)
* ``DMS_bin_score`` / ``DMS_score_bin`` (optional) - binarised label,
  surfaced as ``DMS_bin_score`` when either is present.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


_MUT_RE = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")


def _parse_mutant(token: str) -> Optional[List[Tuple[str, int, str]]]:
    """Return a list of ``(from_aa, pos, to_aa)`` for a colon-separated mutant.

    Returns ``None`` if the token cannot be parsed — those rows are dropped.
    """
    if not isinstance(token, str) or not token:
        return None
    pieces = []
    for piece in token.split(":"):
        m = _MUT_RE.match(piece.strip())
        if not m:
            return None
        src, pos, dst = m.group(1), int(m.group(2)), m.group(3)
        pieces.append((src, pos, dst))
    return pieces


def _revert_mutation(
    mutated_sequence: str, mutations: List[Tuple[str, int, str]]
) -> Optional[str]:
    """Apply the inverse of ``mutations`` to ``mutated_sequence``.

    Returns ``None`` when the stated mutation does not match the
    sequence at the given position (corrupt row).
    """
    chars = list(mutated_sequence)
    for src, pos, dst in mutations:
        idx = pos - 1  # mutant strings use 1-based positions
        if idx < 0 or idx >= len(chars):
            return None
        if chars[idx] != dst:
            return None
        chars[idx] = src
    return "".join(chars)


def _reconstruct_wildtype(df: pd.DataFrame) -> Optional[str]:
    """Pick any usable row and revert its mutation(s) to the wildtype."""
    for _, row in df.iterrows():
        parsed = _parse_mutant(str(row.get("mutant", "")))
        if parsed is None:
            continue
        wt = _revert_mutation(str(row["mutated_sequence"]), parsed)
        if wt is None:
            continue
        return wt
    return None


def load_proteingym(path: Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "mutated_sequence" not in df.columns or "DMS_score" not in df.columns:
        raise ValueError(
            f"ProteinGym file missing required columns: {sorted(set(['mutated_sequence', 'DMS_score']) - set(df.columns))}. "
            f"Available: {list(df.columns)}"
        )

    if "wildtype_sequence" not in df.columns:
        if "mutant" not in df.columns:
            raise ValueError(
                f"ProteinGym file {csv_path} has neither wildtype_sequence nor "
                "a mutant column; cannot reconstruct the wildtype. Available: "
                f"{list(df.columns)}"
            )
        wildtype = _reconstruct_wildtype(df)
        if wildtype is None:
            raise ValueError(
                f"Could not reconstruct wildtype from any row in {csv_path}. "
                "mutant strings may use an unsupported format."
            )
        df = df.copy()
        df["wildtype_sequence"] = wildtype
        logger.info(
            "Reconstructed wildtype for %s (length %d)",
            csv_path.name,
            len(wildtype),
        )

    if "DMS_bin_score" not in df.columns and "DMS_score_bin" in df.columns:
        df = df.rename(columns={"DMS_score_bin": "DMS_bin_score"})

    required = {"mutated_sequence", "wildtype_sequence", "DMS_score"}
    df = df.dropna(subset=list(required)).reset_index(drop=True)
    logger.info("Loaded %d ProteinGym variants from %s", len(df), csv_path)
    return df
