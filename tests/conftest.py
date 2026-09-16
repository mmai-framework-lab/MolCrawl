"""
Conftest for pytest configuration and shared fixtures.
"""

import os
import tempfile

import pytest


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_dna_sequence():
    """Provide a sample DNA sequence for testing."""
    return "ATCGATCGATCGATCG"


@pytest.fixture
def sample_protein_sequence():
    """Provide a sample protein sequence for testing."""
    return "MKTIIALSYIFCLVFADYKDDDDK"


@pytest.fixture
def sample_rna_sequence():
    """Provide a sample RNA sequence for testing."""
    return "AUCGAUCGAUCGAUCG"


@pytest.fixture
def sample_smiles():
    """Provide a sample SMILES string for testing."""
    return "CCO"  # Ethanol


@pytest.fixture
def sample_vocab_file():
    """Create a sample vocab file for testing."""
    vocab_content = """[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
C
c
N
n
O
o
S
s
P
p
F
Cl
Br
I
(
)
[
]
=
#
-
+
1
2
3
4
5
6
7
8
9
0
"""

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(vocab_content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_smiles_data():
    """Provide sample SMILES data for testing."""
    return {
        "valid": [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CC(C)C",
            "C1=CC=C(C=C1)O",
        ],
        "invalid": [
            "",
            ".",
            "INVALID",
            "C(C(C",
        ],
        "complex": [
            "C1=CC=C(C=C1)C(=O)O",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        ],
    }


@pytest.fixture
def mock_compounds_dataset(tmp_path):
    """Create a mock compounds dataset for testing."""
    import pandas as pd

    data = {
        "smiles": [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "INVALID",
            "CC(C)C",
        ],
        "label": [0, 1, 0, 1, 0],
    }

    df = pd.DataFrame(data)
    file_path = tmp_path / "test_compounds.csv"
    df.to_csv(file_path, index=False)

    return file_path


def validate_smiles_output(smiles: str) -> bool:
    """Helper function to validate the validity of generated SMILES."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def calculate_smiles_metrics(generated_smiles: list, reference_smiles: list = None) -> dict:
    """Calculate quality metrics for generated SMILES."""
    valid_count = sum(1 for s in generated_smiles if validate_smiles_output(s))
    total_count = len(generated_smiles)

    metrics = {
        "total": total_count,
        "valid": valid_count,
        "validity_rate": valid_count / total_count if total_count > 0 else 0,
    }

    return metrics


@pytest.fixture
def mock_model_config():
    """Provide a mock model configuration."""
    return {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 256,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "phase1: Phase 1 verification tests")
    config.addinivalue_line("markers", "phase2: Phase 2 dataset tests")
    config.addinivalue_line("markers", "phase3: Phase 3 evaluation tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "data: Tests requiring datasets")
    config.addinivalue_line("markers", "compound: Compounds tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")
    config.addinivalue_line("markers", "bert: BERT tests")
    config.addinivalue_line("markers", "gpt2: GPT2 tests")


try:  # the plugin is not in this environment; see the fixture below
    import pytest_benchmark  # noqa: F401
except ImportError:

    @pytest.fixture
    def benchmark():
        """Stand in for pytest-benchmark's fixture when the plugin is not installed.

        Both users of it -- the two performance tests in tests/unit/test_compounds.py --
        skip on their first line, but a fixture is resolved before the body runs, so
        without the plugin they ended the run as errors rather than skips. Two permanent
        errors in every run are two places a new one can hide, and "303 passed, 2 errors"
        stops meaning anything.

        Defined only when the import fails, so installing pytest-benchmark later gives
        the real fixture back without touching this.
        """
        pytest.skip("pytest-benchmark is not installed")
