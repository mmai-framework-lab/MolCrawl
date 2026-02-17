"""
Conftest for pytest configuration and shared fixtures.
"""

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
