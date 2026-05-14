"""Build ChemLLMBench JSONL files from the upstream CSV / NPZ releases.

The downloader at ``workflows/data/eval-data-chemllmbench.sh`` fetches
the per-task source files from
``https://github.com/ChemFoundationModels/ChemLLMBench/tree/main/data/<task>/``.
This script consumes them and emits the
``{"prompt": ..., "answer": ..., "metadata": {...}}`` JSONL the
evaluator's :func:`load_jsonl` reader expects.

Supported sub-tasks (all backed by upstream files in the public GitHub
repo — no Box / external links needed):

- ``molecule_captioning`` — ``molecule_captioning/molecule_captioning_test.csv``
  schema: ``SMILES``, ``description``
- ``molecule_design`` — ``molecule_design/molecule_design_test.csv``
  schema: ``description``, ``SMILES``
- ``reaction_prediction`` — ``reaction_prediction/uspto_test.csv``
  schema: ``reactant``, ``product``
- ``name_conversion`` — ``name_prediction/llm_test.csv``
  schema: ``CID, smiles, iupac, formula, mol_length, label`` →
  prompt asks for IUPAC given SMILES (the dominant direction in the
  upstream paper).
- ``retrosynthesis`` — ``retro/uspto50k_retro_test.csv``
  schema: ``products_smiles, reactants_smiles``
- ``yield_prediction`` — ``yield_prediction/BH_sample_100_test.npz`` (or
  ``SU_sample_100_test.npz``), expects an ``allow_pickle`` numpy archive
  whose ``data_df`` key holds an ``(N, 2)`` object array of
  ``(reaction_smiles, "Yes"|"No")`` rows.
- ``property_prediction`` — a *directory* containing one or more of
  ``BBBP_test.csv``, ``BACE_test.csv``, ``ClinTox_test.csv``,
  ``HIV_test.csv``, ``Tox_test.csv``. The converter unions every
  matching file it finds, with a ``dataset`` key in metadata so the
  per-row breakdown survives.

Sub-tasks not yet supported here: ``text_guided_generation`` and
``smiles_understanding`` — neither has an obvious upstream artefact in
``ChemFoundationModels/ChemLLMBench/data``. Add a converter once the
data location is identified.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Per-task converters
# ---------------------------------------------------------------------


def _captioning_rows(csv_path: Path) -> Iterable[Tuple[str, str, dict]]:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        smi = str(row["SMILES"]).strip()
        cap = str(row["description"]).strip()
        prompt = f"Describe the following molecule:\n{smi}"
        yield prompt, cap, {"smiles": smi}


def _design_rows(csv_path: Path) -> Iterable[Tuple[str, str, dict]]:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        cap = str(row["description"]).strip()
        smi = str(row["SMILES"]).strip()
        prompt = f"Generate a SMILES that matches:\n{cap}"
        yield prompt, smi, {"description": cap}


def _reaction_rows(csv_path: Path) -> Iterable[Tuple[str, str, dict]]:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        reactant = str(row["reactant"]).strip()
        product = str(row["product"]).strip()
        prompt = f"Predict the product of:\n{reactant}"
        yield prompt, product, {"reactant": reactant}


def _name_conversion_rows(csv_path: Path) -> Iterable[Tuple[str, str, dict]]:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        smi = str(row["smiles"]).strip()
        iupac = str(row["iupac"]).strip()
        if not smi or not iupac or iupac.lower() == "nan":
            continue
        prompt = f"What is the IUPAC name of the molecule with SMILES:\n{smi}"
        meta = {"smiles": smi, "cid": int(row["CID"]) if "CID" in row and pd.notna(row["CID"]) else None}
        yield prompt, iupac, {k: v for k, v in meta.items() if v is not None}


def _retrosynthesis_rows(csv_path: Path) -> Iterable[Tuple[str, str, dict]]:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        product = str(row["products_smiles"]).strip()
        reactants = str(row["reactants_smiles"]).strip()
        if not product or not reactants:
            continue
        prompt = (
            "Predict the reactants required to synthesise the following "
            f"product. Return SMILES of all reactants joined with '.'.\n"
            f"Product: {product}"
        )
        yield prompt, reactants, {"product": product}


def _yield_prediction_rows(npz_path: Path) -> Iterable[Tuple[str, str, dict]]:
    """Read a ChemLLMBench yield_prediction NPZ.

    Each NPZ stores a single ``data_df`` key holding an ``(N, 2)``
    object array of ``[reaction_smiles, "Yes"|"No"]`` pairs.
    """
    import numpy as np

    archive = np.load(npz_path, allow_pickle=True)
    if "data_df" not in archive.files:
        raise ValueError(
            f"NPZ {npz_path} missing 'data_df' key (have: {archive.files})"
        )
    data = archive["data_df"]
    for i in range(data.shape[0]):
        rxn = str(data[i, 0]).strip()
        label = str(data[i, 1]).strip()
        if not rxn or label not in {"Yes", "No"}:
            continue
        prompt = (
            "You are an expert chemist. Decide whether the following "
            "reaction's yield exceeds 30%. Reply with exactly 'Yes' or 'No'.\n"
            "Reaction: " + rxn
        )
        yield prompt, label, {"reaction": rxn, "source": npz_path.name}


# Property prediction wraps multiple per-dataset CSVs into one task.
_PROPERTY_DATASETS = {
    # filename-stem (lower) → (smiles_column, label_column,
    #   prompt_question, label_to_answer_map)
    "bbbp_test": (
        "smiles", "p_np",
        "Does this molecule penetrate the blood-brain barrier? "
        "Reply with exactly 'Yes' or 'No'.",
        {"1": "Yes", "0": "No"},
    ),
    "bace_test": (
        "mol", "Class",
        "Is this molecule an inhibitor of human β-secretase 1 (BACE-1)? "
        "Reply with exactly 'Yes' or 'No'.",
        {"1": "Yes", "0": "No"},
    ),
    "clintox_test": (
        "smiles", "FDA_APPROVED",
        "Is this molecule FDA-approved? "
        "Reply with exactly 'Yes' or 'No'.",
        {"Yes": "Yes", "No": "No", "1": "Yes", "0": "No"},
    ),
    "hiv_test": (
        "smiles", "HIV_active",
        "Is this molecule active against HIV? "
        "Reply with exactly 'Yes' or 'No'.",
        {"1": "Yes", "0": "No"},
    ),
    "tox_test": (
        "smiles", "NR-AR",
        "Is this molecule active in the NR-AR androgen-receptor toxicity "
        "assay? Reply with exactly 'Yes' or 'No'.",
        {"1.0": "Yes", "0.0": "No", "1": "Yes", "0": "No"},
    ),
}


def _property_prediction_rows(source_dir: Path) -> Iterable[Tuple[str, str, dict]]:
    """Read every *_test.csv under ``source_dir`` matching a known schema."""
    if not source_dir.is_dir():
        raise ValueError(
            f"property_prediction expects a directory, got file: {source_dir}"
        )
    csv_files = sorted(source_dir.glob("*_test.csv"))
    if not csv_files:
        raise ValueError(
            f"No '*_test.csv' files under {source_dir}; "
            "expected one or more of: BBBP_test.csv, BACE_test.csv, "
            "ClinTox_test.csv, HIV_test.csv, Tox_test.csv"
        )
    for csv_path in csv_files:
        stem = csv_path.stem.lower()
        if stem not in _PROPERTY_DATASETS:
            logger.info("property_prediction: skipping unrecognised %s", csv_path.name)
            continue
        smiles_col, label_col, question, label_map = _PROPERTY_DATASETS[stem]
        df = pd.read_csv(csv_path)
        if smiles_col not in df.columns or label_col not in df.columns:
            logger.warning(
                "property_prediction: %s missing required cols (have %s); skipping",
                csv_path.name, list(df.columns),
            )
            continue
        for _, row in df.iterrows():
            smi = str(row[smiles_col]).strip()
            raw = str(row[label_col]).strip()
            if not smi or smi.lower() == "nan":
                continue
            answer = label_map.get(raw)
            if answer is None:
                # row's label is missing or in an unexpected format — skip
                continue
            prompt = f"{question}\nSMILES: {smi}"
            yield prompt, answer, {"smiles": smi, "dataset": stem.replace("_test", "")}


_CONVERTERS = {
    "molecule_captioning": _captioning_rows,
    "molecule_design": _design_rows,
    "reaction_prediction": _reaction_rows,
    "name_conversion": _name_conversion_rows,
    "retrosynthesis": _retrosynthesis_rows,
    "yield_prediction": _yield_prediction_rows,
    "property_prediction": _property_prediction_rows,
}


def convert_csv_to_jsonl(
    task: str,
    csv_path: Path,
    output_jsonl: Path,
    max_examples: Optional[int] = None,
) -> dict:
    if task not in _CONVERTERS:
        raise ValueError(
            f"Unsupported sub-task {task!r}. Currently supported: "
            f"{list(_CONVERTERS.keys())}. Two of the nine ChemLLMBench "
            "sub-tasks (text_guided_generation, smiles_understanding) "
            "have no obvious upstream artefact yet; flag them when their "
            "data location is identified."
        )
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for prompt, answer, meta in _CONVERTERS[task](csv_path):
            if max_examples is not None and n_written >= max_examples:
                break
            fh.write(
                json.dumps(
                    {"prompt": prompt, "answer": answer, "metadata": dict(meta)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1
    logger.info("Wrote %d examples for task=%r -> %s", n_written, task, output_jsonl)
    return {"task": task, "n_examples": n_written, "output": str(output_jsonl)}


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Convert ChemLLMBench upstream CSVs into the {prompt, answer, metadata} JSONL"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(_CONVERTERS.keys()),
        help="Sub-task name; only the in-repo CSV-shipping ones are wired here.",
    )
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args(argv)

    convert_csv_to_jsonl(
        task=args.task,
        csv_path=Path(args.source_csv),
        output_jsonl=Path(args.output_jsonl),
        max_examples=args.max_examples,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
