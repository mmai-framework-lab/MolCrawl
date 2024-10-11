from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def prepare_scaffolds(smiles: str):
    """
    Prepare the scaffolds of a molecule.
    """
    if smiles == ".":
        return ""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    
    scaffold = GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles
