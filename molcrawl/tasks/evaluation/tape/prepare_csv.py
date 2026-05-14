"""Materialise TAPE sub-task JSONL files from public HuggingFace mirrors.

The original TAPE release (songlab-cal/tape) hosts its data on the
``songlabdata`` S3 bucket, which has returned 403 anonymously since
2025. Several community mirrors keep the same data publicly available:

- ``AI4Protein/TAPE_Fluorescence`` — fluorescence regression
- ``AI4Protein/TAPE_Stability``    — stability regression
- ``proteinea/remote_homology``    — remote homology classification (1195 folds)
- ``proteinea/secondary_structure_prediction`` — SS3 / SS8 sequence labelling
- ``proteinglm/contact_prediction_binary`` — residue-residue contact pairs

We fetch the right file from the right mirror, rename columns to the
canonical TAPE schema (``primary`` for sequence, task-specific label
column matching :class:`TAPETaskSpec`), and emit
``<task>_<split>.json`` JSONL files in the layout the existing
:func:`splits.load_splits` expects.

All six TAPE sub-tasks are now wired through public mirrors.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _write_jsonl(records: List[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def _materialise_fluorescence(output_dir: Path) -> Dict[str, int]:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    counts: Dict[str, int] = {}
    for upstream_split, our_split in (
        ("train.csv", "train"),
        ("valid.csv", "valid"),
        ("test.csv", "test"),
    ):
        path = hf_hub_download(
            "AI4Protein/TAPE_Fluorescence", upstream_split, repo_type="dataset"
        )
        df = pd.read_csv(path)
        df = df.rename(columns={"aa_seq": "primary", "label": "log_fluorescence"})
        records = df[["primary", "log_fluorescence"]].dropna().to_dict(orient="records")
        out = output_dir / "fluorescence" / f"fluorescence_{our_split}.json"
        counts[our_split] = _write_jsonl(records, out)
        logger.info(
            "fluorescence/%s: %d records -> %s", our_split, counts[our_split], out
        )
    return counts


def _materialise_stability(output_dir: Path) -> Dict[str, int]:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    counts: Dict[str, int] = {}
    for upstream_split, our_split in (
        ("train.csv", "train"),
        ("valid.csv", "valid"),
        ("test.csv", "test"),
    ):
        path = hf_hub_download(
            "AI4Protein/TAPE_Stability", upstream_split, repo_type="dataset"
        )
        df = pd.read_csv(path)
        df = df.rename(columns={"aa_seq": "primary", "label": "stability_score"})
        records = df[["primary", "stability_score"]].dropna().to_dict(orient="records")
        out = output_dir / "stability" / f"stability_{our_split}.json"
        counts[our_split] = _write_jsonl(records, out)
        logger.info(
            "stability/%s: %d records -> %s", our_split, counts[our_split], out
        )
    return counts


def _materialise_remote_homology(output_dir: Path) -> Dict[str, int]:
    """Classification on ``fold_label`` with 1195 fold classes.

    Upstream release ships THREE held-out test splits (family / superfamily /
    fold). We materialise all of them but the evaluator pipeline only
    consumes ``train`` + ``valid`` + ``test``; we map the
    fold-holdout split (the hardest = most-OOD setting) to ``test``.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    counts: Dict[str, int] = {}
    file_map = {
        "train.csv": "train",
        "valid.csv": "valid",
        "test_fold_holdout.csv": "test",
    }
    for upstream_name, our_split in file_map.items():
        path = hf_hub_download(
            "proteinea/remote_homology", upstream_name, repo_type="dataset"
        )
        df = pd.read_csv(path)
        keep = ["primary", "fold_label"]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"proteinea/remote_homology/{upstream_name} missing columns {missing}; "
                f"got {list(df.columns)}"
            )
        records = df[keep].dropna().to_dict(orient="records")
        out = output_dir / "remote_homology" / f"remote_homology_{our_split}.json"
        counts[our_split] = _write_jsonl(records, out)
        logger.info(
            "remote_homology/%s: %d records -> %s",
            our_split,
            counts[our_split],
            out,
        )
    return counts


_DSSP3_ALPHABET = ("C", "E", "H")
# Canonical 8-class DSSP alphabet. Upstream proteinea CSVs use an
# 8-letter subset; we order them deterministically so per-residue label
# integers are stable across runs.
_DSSP8_ALPHABET = ("B", "C", "E", "G", "H", "I", "S", "T")


def _parse_mask(value: str) -> List[float]:
    """Parse the 1.0/0.0 mask strings shipped by proteinea.

    Different upstream rows use different formats: space-separated
    (``"1.0 1.0 0.0"``), comma-separated (``"1.0,0.0,1.0"``), or
    Python-list-as-string (``"[1.0, 0.0, 1.0]"``). Normalise all to a
    flat list of floats.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    s = str(value).strip()
    if not s:
        return []
    # strip Python-list brackets if present
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # unify separators: replace commas with spaces
    s = s.replace(",", " ")
    out: List[float] = []
    for token in s.split():
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            # skip stray non-numeric tokens (e.g. ``nan``, empty cells)
            continue
    return out


def _encode_label_chars(s: str, alphabet: tuple) -> List[int]:
    """Map ``CHHHEE...`` to integer ids; chars outside the alphabet → -1."""
    lookup = {c: i for i, c in enumerate(alphabet)}
    return [int(lookup.get(c, -1)) for c in str(s)]


def _materialise_secondary_structure(
    output_dir: Path, k: int
) -> Dict[str, int]:
    """Per-residue secondary-structure prediction (DSSP-3 or DSSP-8).

    Source: proteinea/secondary_structure_prediction. Upstream ships
    ``training_hhblits.csv`` (10,792 proteins) plus per-CASP test
    splits. We map:

      * ``training_hhblits.csv`` -> our ``train`` (carved into train+valid
        with the seed=42 random split done by the evaluator's loader).
      * ``CB513.csv`` -> ``test`` (the canonical SS benchmark).

    Other test splits (CASP12 / CASP13 / CASP14 / TS115) are written
    alongside as ``<task>_<casp_split>.json`` for ad-hoc analysis but
    the evaluator only consumes ``train`` + ``test``.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    alphabet: Tuple[str, ...]
    if k == 3:
        task = "secondary_structure_3"
        label_col = "ss3"
        upstream_label_col = "dssp3"
        alphabet = _DSSP3_ALPHABET
    elif k == 8:
        task = "secondary_structure_8"
        label_col = "ss8"
        upstream_label_col = "dssp8"
        alphabet = _DSSP8_ALPHABET
    else:
        raise ValueError(f"unsupported k={k!r}; choose 3 or 8")

    upstream_to_split = {
        "training_hhblits.csv": "train",
        "CB513.csv": "test",
    }
    extra_test_csvs = ["CASP12.csv", "CASP13.csv", "CASP14.csv", "TS115.csv"]

    counts: Dict[str, int] = {}
    for upstream_name, our_split in upstream_to_split.items():
        path = hf_hub_download(
            "proteinea/secondary_structure_prediction",
            upstream_name,
            repo_type="dataset",
        )
        df = pd.read_csv(path)
        records = []
        for _, row in df.iterrows():
            primary = str(row["input"])
            label_str = str(row.get(upstream_label_col, ""))
            if len(primary) != len(label_str):
                # rare upstream rows where the lengths disagree — skip rather than
                # corrupt the per-residue alignment.
                continue
            labels = _encode_label_chars(label_str, alphabet)
            disorder = _parse_mask(row.get("disorder", ""))
            # ``disorder`` is 1.0 for ordered (use this residue) and 0.0 for
            # disordered (skip); the evaluator masks at 0 if either disorder
            # is unset or 1.0.
            valid_mask = [1 if (i >= len(disorder) or disorder[i] >= 0.5) else 0
                          for i in range(len(primary))]
            # also drop residues whose label fell outside the alphabet
            valid_mask = [
                1 if (m and labels[i] >= 0) else 0
                for i, m in enumerate(valid_mask)
            ]
            records.append(
                {
                    "primary": primary,
                    label_col: labels,
                    f"{label_col}_str": label_str,
                    "valid_mask": valid_mask,
                }
            )
        out = output_dir / task / f"{task}_{our_split}.json"
        counts[our_split] = _write_jsonl(records, out)
        logger.info("%s/%s: %d records -> %s", task, our_split, counts[our_split], out)

    # Optional extra CASP test splits (no canonical mapping — keep available)
    for name in extra_test_csvs:
        try:
            path = hf_hub_download(
                "proteinea/secondary_structure_prediction",
                name,
                repo_type="dataset",
            )
        except Exception as exc:  # pragma: no cover
            logger.info("Skipping optional %s/%s: %s", task, name, exc)
            continue
        df = pd.read_csv(path)
        records = []
        for _, row in df.iterrows():
            primary = str(row["input"])
            label_str = str(row.get(upstream_label_col, ""))
            if len(primary) != len(label_str):
                continue
            labels = _encode_label_chars(label_str, alphabet)
            disorder = _parse_mask(row.get("disorder", ""))
            valid_mask = [
                1 if (i >= len(disorder) or disorder[i] >= 0.5) else 0
                for i in range(len(primary))
            ]
            valid_mask = [
                1 if (m and labels[i] >= 0) else 0
                for i, m in enumerate(valid_mask)
            ]
            records.append(
                {
                    "primary": primary,
                    label_col: labels,
                    f"{label_col}_str": label_str,
                    "valid_mask": valid_mask,
                }
            )
        suffix = name.replace(".csv", "").lower()  # e.g. "casp12"
        out = output_dir / task / f"{task}_{suffix}.json"
        counts[suffix] = _write_jsonl(records, out)
        logger.info("%s/%s: %d records -> %s", task, suffix, counts[suffix], out)
    return counts


def _materialise_secondary_structure_3(output_dir: Path) -> Dict[str, int]:
    return _materialise_secondary_structure(output_dir, k=3)


def _materialise_secondary_structure_8(output_dir: Path) -> Dict[str, int]:
    return _materialise_secondary_structure(output_dir, k=8)


def _materialise_contact_prediction(output_dir: Path) -> Dict[str, int]:
    """Materialise contact_prediction from ``proteinglm/contact_prediction_binary``.

    The mirror ships parquet files (``data/{train,valid,test}-00000-of-00001.parquet``)
    with two columns: ``seq`` (the amino-acid sequence) and ``label`` (a 1-D
    object array of ``np.array([i, j])`` index pairs that ARE in contact —
    the negative pairs are implicit in the complement). We emit one JSONL
    record per protein with the sequence under ``primary`` and the contact
    pair list under ``tertiary`` (the column name :class:`TAPETaskSpec`
    points at — kept for backward compat with the upstream TAPE schema even
    though the actual content is a contact list, not raw 3-D coordinates).
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd

    counts: Dict[str, int] = {}
    for upstream, our_split in (
        ("data/train-00000-of-00001.parquet", "train"),
        ("data/valid-00000-of-00001.parquet", "valid"),
        ("data/test-00000-of-00001.parquet", "test"),
    ):
        path = hf_hub_download(
            "proteinglm/contact_prediction_binary",
            upstream,
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        records: List[dict] = []
        for _, row in df.iterrows():
            seq = str(row["seq"])
            label = row["label"]
            try:
                pairs = [[int(p[0]), int(p[1])] for p in label if len(p) >= 2]
            except (TypeError, ValueError):
                pairs = []
            if not seq or not pairs:
                continue
            records.append({"primary": seq, "tertiary": pairs})
        out = output_dir / "contact_prediction" / f"contact_prediction_{our_split}.json"
        counts[our_split] = _write_jsonl(records, out)
        logger.info(
            "contact_prediction/%s: %d records -> %s",
            our_split,
            counts[our_split],
            out,
        )
    return counts


_MATERIALISERS = {
    "fluorescence": _materialise_fluorescence,
    "stability": _materialise_stability,
    "remote_homology": _materialise_remote_homology,
    "secondary_structure_3": _materialise_secondary_structure_3,
    "secondary_structure_8": _materialise_secondary_structure_8,
    "contact_prediction": _materialise_contact_prediction,
}


def materialise_tape(
    output_dir: Path, tasks: Optional[List[str]] = None
) -> Dict[str, Dict[str, int]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = tasks if tasks else list(_MATERIALISERS.keys())
    summary: Dict[str, Dict[str, int]] = {}
    for t in selected:
        if t not in _MATERIALISERS:
            logger.warning(
                "task %r has no public-mirror materialiser; skipping. "
                "Available: %s",
                t,
                list(_MATERIALISERS.keys()),
            )
            summary[t] = {}
            continue
        summary[t] = _MATERIALISERS[t](output_dir)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Materialise TAPE JSONL splits from public HF mirrors"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Tasks to materialise (default: all wired = {list(_MATERIALISERS.keys())})",
    )
    args = parser.parse_args(argv)

    materialise_tape(output_dir=Path(args.output_dir), tasks=args.tasks)


if __name__ == "__main__":  # pragma: no cover
    main()
