"""Per-generated-SMILES dumps for the MOSES evaluator.

Mirrors the per-row prediction logs of the variant-effect evaluators
but the unit is one *generated* molecule rather than a ground-truth
row. ``predictions.jsonl`` writes one record per molecule with its
canonical form, validity, novelty, and a small property summary;
``predictions.txt`` previews molecules across the {valid+novel,
valid+seen, invalid} quadrants so reviewers can see concrete examples
of where the model regurgitates training data, where it discovers new
chemistry, and where it fails outright.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def write_predictions(
    output_dir: Path,
    generated_raw: Sequence[str],
    canonicalised: Sequence[Optional[str]],
    train_canonical: Set[str],
    test_canonical: Optional[Set[str]] = None,
    scaffolds_canonical: Optional[Set[str]] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
    failure_mode_counts: Optional[Dict[str, int]] = None,
    arch: Optional[str] = None,
    preview_count: int = 30,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    properties = _per_molecule_properties(canonicalised)

    jsonl_path = output_dir / "predictions.jsonl"
    _write_jsonl(
        jsonl_path,
        generated_raw=generated_raw,
        canonicalised=canonicalised,
        properties=properties,
        train_canonical=train_canonical,
        test_canonical=test_canonical,
        scaffolds_canonical=scaffolds_canonical,
    )

    txt_path = output_dir / "predictions.txt"
    _write_narrative(
        txt_path,
        generated_raw=generated_raw,
        canonicalised=canonicalised,
        properties=properties,
        train_canonical=train_canonical,
        test_canonical=test_canonical,
        scaffolds_canonical=scaffolds_canonical,
        sampling_params=sampling_params,
        failure_mode_counts=failure_mode_counts,
        arch=arch,
        preview_count=preview_count,
    )

    return {
        "predictions_jsonl": str(jsonl_path),
        "predictions_txt": str(txt_path),
    }


# ---------------------------------------------------------------------
# Per-molecule property extraction (RDKit if available)
# ---------------------------------------------------------------------


def _per_molecule_properties(
    canonicalised: Sequence[Optional[str]],
) -> List[Dict[str, Any]]:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors

        RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

        def props(canon: Optional[str]) -> Dict[str, Any]:
            if canon is None:
                return {}
            mol = Chem.MolFromSmiles(canon)
            if mol is None:
                return {}
            return {
                "atom_count": int(mol.GetNumAtoms()),
                "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
                "ring_count": int(mol.GetRingInfo().NumRings()),
                "mol_weight": float(Descriptors.MolWt(mol)),  # type: ignore[attr-defined]
            }
    except ImportError:
        def props(canon: Optional[str]) -> Dict[str, Any]:  # noqa: ARG001
            return {}

    return [props(c) for c in canonicalised]


# ---------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------


def _write_jsonl(
    path: Path,
    generated_raw: Sequence[str],
    canonicalised: Sequence[Optional[str]],
    properties: Sequence[Dict[str, Any]],
    train_canonical: Set[str],
    test_canonical: Optional[Set[str]],
    scaffolds_canonical: Optional[Set[str]],
) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, (raw, canon, prop) in enumerate(
            zip(generated_raw, canonicalised, properties)
        ):
            record: Dict[str, Any] = {
                "index": int(i),
                "generated_raw": str(raw),
                "canonical": canon,
                "valid": canon is not None,
                "novel_vs_train": (
                    None if canon is None else (canon not in train_canonical)
                ),
                "novel_vs_test": (
                    None
                    if (canon is None or test_canonical is None)
                    else (canon not in test_canonical)
                ),
                "novel_vs_scaffolds": (
                    None
                    if (canon is None or scaffolds_canonical is None)
                    else (canon not in scaffolds_canonical)
                ),
            }
            record.update(prop)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------


_SECTION = "=" * 72
_RULE = "-" * 72


def _write_narrative(
    path: Path,
    generated_raw: Sequence[str],
    canonicalised: Sequence[Optional[str]],
    properties: Sequence[Dict[str, Any]],
    train_canonical: Set[str],
    test_canonical: Optional[Set[str]],
    scaffolds_canonical: Optional[Set[str]],
    sampling_params: Optional[Dict[str, Any]],
    failure_mode_counts: Optional[Dict[str, int]],
    arch: Optional[str],
    preview_count: int,
) -> None:
    n = len(generated_raw)
    valid_idx = [i for i, c in enumerate(canonicalised) if c is not None]
    invalid_idx = [i for i, c in enumerate(canonicalised) if c is None]

    valid_novel = [
        i for i in valid_idx if canonicalised[i] not in train_canonical
    ]
    valid_seen = [
        i for i in valid_idx if canonicalised[i] in train_canonical
    ]

    with path.open("w", encoding="utf-8") as fh:
        fh.write("MOSES per-generated-SMILES narrative\n")
        fh.write(_SECTION + "\n")
        fh.write(f"arch            : {arch or '?'}\n")
        fh.write(f"n_generated     : {n}\n")
        fh.write(f"n_valid         : {len(valid_idx)}\n")
        fh.write(f"n_invalid       : {len(invalid_idx)}\n")
        fh.write(f"n_valid_novel   : {len(valid_novel)}  (canonical not in train)\n")
        fh.write(f"n_valid_seen    : {len(valid_seen)}  (canonical found in train)\n")
        if test_canonical is not None:
            valid_test_novel = [
                i for i in valid_idx if canonicalised[i] not in test_canonical
            ]
            fh.write(
                f"n_valid_novel_vs_test : {len(valid_test_novel)}\n"
            )
        if scaffolds_canonical is not None:
            valid_scaf_novel = [
                i for i in valid_idx if canonicalised[i] not in scaffolds_canonical
            ]
            fh.write(
                f"n_valid_novel_vs_scaffolds : {len(valid_scaf_novel)}\n"
            )
        if sampling_params:
            fh.write(f"sampling_params : {sampling_params}\n")
        if failure_mode_counts:
            fh.write("failure_modes   : ")
            for mode, count in sorted(
                failure_mode_counts.items(), key=lambda kv: -kv[1]
            ):
                fh.write(f"{mode}={count} ")
            fh.write("\n")

        # Property summary over the valid subset
        if valid_idx:
            atom_counts = [
                properties[i].get("atom_count")
                for i in valid_idx
                if properties[i].get("atom_count") is not None
            ]
            mol_weights = [
                properties[i].get("mol_weight")
                for i in valid_idx
                if properties[i].get("mol_weight") is not None
            ]
            if atom_counts:
                a = np.array(atom_counts, dtype=float)
                fh.write(
                    f"atom_count      : mean={a.mean():.1f} std={a.std():.1f} "
                    f"min={a.min():.0f} max={a.max():.0f}\n"
                )
            if mol_weights:
                m = np.array(mol_weights, dtype=float)
                fh.write(
                    f"mol_weight (Da) : mean={m.mean():.1f} std={m.std():.1f} "
                    f"min={m.min():.1f} max={m.max():.1f}\n"
                )

        fh.write("\n")
        fh.write(
            "Score blocks below sample three quadrants of generation behaviour: "
            "(A) valid + novel = the model produces new, parseable molecules; "
            "(B) valid + seen  = the model regurgitates training molecules; "
            "(C) invalid       = parser-level failures (the rough cause is in "
            "failure_modes above).\n"
        )
        fh.write(_SECTION + "\n\n")

        per_quadrant = max(1, preview_count // 3)
        rng = np.random.default_rng(seed=0)

        def _draw(pool: Sequence[int], take: int) -> List[int]:
            if not pool:
                return []
            take = min(take, len(pool))
            return [int(x) for x in rng.choice(pool, size=take, replace=False)]

        rank = 0
        for label, pool in [
            ("valid + novel", valid_novel),
            ("valid + seen", valid_seen),
            ("invalid", invalid_idx),
        ]:
            picks = _draw(pool, per_quadrant)
            for i in picks:
                rank += 1
                _write_one_example(
                    fh,
                    rank=rank,
                    label=label,
                    raw=generated_raw[i],
                    canonical=canonicalised[i],
                    prop=properties[i],
                    train_canonical=train_canonical,
                )


def _write_one_example(
    fh,
    rank: int,
    label: str,
    raw: str,
    canonical: Optional[str],
    prop: Dict[str, Any],
    train_canonical: Set[str],
) -> None:
    fh.write(_SECTION + "\n")
    fh.write(f"Example {rank}  ({label})\n")
    fh.write(_RULE + "\n")
    fh.write(f"raw       : {raw}\n")
    fh.write(f"canonical : {canonical if canonical is not None else '(unparseable)'}\n")
    if prop:
        fh.write(
            f"props     : atoms={prop.get('atom_count','?')}  "
            f"heavy={prop.get('heavy_atom_count','?')}  "
            f"rings={prop.get('ring_count','?')}  "
            f"MW={prop.get('mol_weight','?'):.1f}\n"
            if isinstance(prop.get("mol_weight"), (int, float))
            else f"props     : atoms={prop.get('atom_count','?')}\n"
        )
    fh.write(_SECTION + "\n\n")
