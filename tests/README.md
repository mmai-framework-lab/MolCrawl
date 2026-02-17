"""
README for test suite.
"""

# Test Suite Documentation

## Structure

```
tests/
├── conftest.py                # Shared fixtures and configuration
├── unit/                      # Unit tests
│   ├── test_tokenizers.py
│   ├── test_data_utils.py
│   └── test_model_utils.py
├── integration/               # Integration tests
│   ├── test_bert_pipeline.py
│   ├── test_gpt2_pipeline.py
│   └── test_data_loading.py
├── phase1/                    # Phase 1 verification tests
│   ├── test_bert_domains.py
│   └── test_gpt2_domains.py
├── phase2/                    # Phase 2 dataset tests
│   └── test_dataset_preparation.py
├── phase3/                    # Phase 3 evaluation tests
│   └── test_model_evaluation.py
├── benchmarks/                # Performance benchmarks
└── data/                      # Data pipeline tests
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Phase-specific tests
pytest -m phase1
pytest -m phase2
pytest -m phase3

# Domain-specific tests
pytest -m dna
pytest -m protein
pytest -m compound
```

### Run tests with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run benchmarks
```bash
pytest -m benchmark --benchmark-only
```

## Writing Tests

### Naming Convention
- Test files: `test_*.py`
- Test functions: `test_*()`
- Test classes: `Test*`

### Using Markers
```python
import pytest

@pytest.mark.unit
@pytest.mark.dna
def test_dna_tokenization():
    # Your test code here
    pass
```

### Using Fixtures
```python
def test_with_sample_data(sample_dna_sequence):
    # sample_dna_sequence is provided by conftest.py
    assert len(sample_dna_sequence) > 0
```

## Test Status

### Phase 1
- [ ] BERT DNA verification
- [ ] BERT Protein verification
- [ ] BERT RNA verification
- [ ] BERT Compound verification
- [ ] BERT Compound-Lang verification
- [ ] GPT2 DNA verification
- [ ] GPT2 Protein verification
- [ ] GPT2 RNA verification
- [ ] GPT2 Compound verification
- [ ] GPT2 Compound-Lang verification

### Phase 2
- [ ] Dataset preparation tests
- [ ] Training script verification
- [ ] Log management tests

### Phase 3
- [ ] Alpha model evaluation
- [ ] Performance benchmarks
- [ ] Model card generation

## CI Integration

Tests are automatically run in CI on:
- Push to main, develop, or feature branches
- Pull requests
- Manual workflow dispatch

See `.github/workflows/ci-tests.yml` for details.
