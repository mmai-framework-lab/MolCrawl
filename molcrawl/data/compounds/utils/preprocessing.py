from __future__ import annotations

import logging
from typing import List, Tuple

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
except ModuleNotFoundError:
    Chem = None
    RDLogger = None
    GetScaffoldForMol = None

# Suppress RDKit warnings (because there are many warnings about invalid SMILES structures)
# However, errors are logged
if RDLogger is not None:
    rdkit_logger = RDLogger.logger()
    rdkit_logger.setLevel(RDLogger.ERROR)

logger = logging.getLogger(__name__)

# keep statistics for disabled SMILES
_invalid_smiles_count = 0
_total_smiles_count = 0
_invalid_smiles_examples: List[Tuple[str, str]] = []  # Save invalid SMILES examples (up to 10)


def get_invalid_smiles_stats():
    """
    Get statistics for invalid SMILES

        Returns:
    tuple: (number of invalid SMILES, total number of SMILES, ineffectiveness, list of invalid examples)
    """
    if _total_smiles_count == 0:
        return 0, 0, 0.0, []
    invalid_rate = (_invalid_smiles_count / _total_smiles_count) * 100
    return _invalid_smiles_count, _total_smiles_count, invalid_rate, _invalid_smiles_examples


def prepare_scaffolds(smiles: str):
    """
        Prepare the scaffolds of a molecule.

        Args:
    smiles: SMILES string

        Returns:
    str: scaffold SMILES string, empty string if invalid

        Note:
    Large databases such as ZINC20 may contain invalid SMILES for the following reasons:
    1. Notation problems for ionic structures such as quaternary ammonium (N+)
    2. Conversion error from different formats
    3. Expression of special stereochemistry
    4. Automatic processing error when creating database

    These are usually within an acceptable range of a few percent of the total database.
    """
    global _invalid_smiles_count, _total_smiles_count, _invalid_smiles_examples
    _total_smiles_count += 1

    if smiles == "." or not smiles:
        _invalid_smiles_count += 1
        if len(_invalid_smiles_examples) < 10:
            _invalid_smiles_examples.append(("empty or dot", smiles))
        return ""

    if Chem is None or GetScaffoldForMol is None:
        raise ModuleNotFoundError("rdkit is required to prepare scaffolds")

    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            _invalid_smiles_count += 1
            # save first 10 invalid examples
            if len(_invalid_smiles_examples) < 10:
                _invalid_smiles_examples.append(("parse_failed", smiles[:100]))

            # Log statistics every 1000 items
            if _invalid_smiles_count % 1000 == 0:
                invalid_count, total_count, invalid_rate, examples = get_invalid_smiles_stats()
                logger.warning(f"Invalid SMILES detected: {invalid_count}/{total_count} ({invalid_rate:.2f}%)")
            return ""

        scaffold = GetScaffoldForMol(molecule)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    except Exception as e:
        _invalid_smiles_count += 1
        if len(_invalid_smiles_examples) < 10:
            _invalid_smiles_examples.append(("exception", f"{smiles[:100]} | Error: {str(e)[:50]}"))
        logger.debug(f"Error processing SMILES '{smiles[:50]}...': {e}")
        return ""
