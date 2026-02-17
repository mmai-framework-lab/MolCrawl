# CI/CD Configuration Documentation

## Overview
This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the RIKEN Dataset Foundation Model project.

## Workflow Files

### 1. `ci-tests.yml` - Continuous Integration Tests
**Triggers**: Push to main/develop/feature branches, Pull Requests  
**Purpose**: Ensures code quality and functionality

**Jobs**:
- **unit-tests**: Runs unit tests with pytest (Python 3.9, 3.10)
- **integration-tests**: Tests component integration
- **model-sanity-checks**: Validates BERT and GPT2 model initialization
- **data-pipeline-validation**: Checks data loading and processing
- **type-checking**: Static type analysis with MyPy
- **security-scan**: Security vulnerability scanning with Safety and Bandit

### 2. `phase-validation.yml` - Phase-Specific Validation
**Triggers**: Manual workflow dispatch  
**Purpose**: Validates completion of specific project phases

**Workflows**:
- **Phase 1 BERT Verification**: Tests BERT models across all domains
- **Phase 1 GPT2 Verification**: Tests GPT2 models across all domains
- **Phase 2 Dataset Preparation**: Validates dataset creation scripts
- **Phase 2 Script Verification**: Verifies training script functionality
- **Phase 3 Alpha Evaluation**: Comprehensive model evaluation

### 3. `documentation.yml` - Documentation Generation
**Triggers**: Push to main/develop, Manual dispatch  
**Purpose**: Builds and deploys documentation

**Jobs**:
- **build-docs**: Generates Sphinx documentation
- **check-readme**: Validates markdown formatting and links
- **generate-api-docs**: Creates API reference with pdoc

### 4. `benchmark.yml` - Performance Benchmarks
**Triggers**: Weekly schedule (Sunday 00:00 UTC), Manual dispatch  
**Purpose**: Tracks model and pipeline performance over time

**Jobs**:
- **benchmark-models**: Tests model inference performance
- **data-pipeline-benchmark**: Measures data loading efficiency

### 5. `release.yml` - Release Process
**Triggers**: Version tags (v*.*.*, alpha-*, beta-*), Manual dispatch  
**Purpose**: Automates release preparation and distribution

**Jobs**:
- **validate-release**: Runs comprehensive tests
- **build-package**: Creates distribution packages
- **prepare-huggingface**: Generates model cards for HF Hub
- **create-release**: Creates GitHub release
- **notify-release**: Sends release notifications

## Phase-Specific CI Strategy

### Phase 1: CBI Conference - Functional Verification
**CI Focus**: Model functionality validation
- ✅ Ruff linting (already configured)
- ✅ ESLint for web components
- ✅ Unit tests for core functions
- ✅ Model initialization checks
- Manual phase validation workflow for each task

**Usage**:
```bash
# Trigger BERT verification for all domains
gh workflow run phase-validation.yml -f phase=phase1-bert-verification

# Trigger GPT2 verification
gh workflow run phase-validation.yml -f phase=phase1-gpt2-verification
```

### Phase 2: Pre-alpha - Dataset Preparation and Method Preparation
**CI Focus**: Data pipeline and training script validation
- ✅ Dataset loading tests
- ✅ Training script syntax validation
- ✅ Benchmark data configuration checks
- ✅ Integration tests for data pipelines

**Usage**:
```bash
# Validate dataset preparation
gh workflow run phase-validation.yml -f phase=phase2-dataset-prep

# Verify training scripts
gh workflow run phase-validation.yml -f phase=phase2-script-verification
```

### Phase 3: Alpha - Hyperparameter Tuning
**CI Focus**: Model performance and regression testing
- ✅ Model evaluation benchmarks
- ✅ Performance regression detection
- ✅ Training log validation
- ✅ Model card generation
- ✅ Release preparation automation

**Usage**:
```bash
# Run alpha model evaluation
gh workflow run phase-validation.yml -f phase=phase3-alpha-evaluation

# Run performance benchmarks
gh workflow run benchmark.yml
```

### Phase 4: Paper Writing
**CI Focus**: Documentation and reproducibility
- ✅ Automated documentation building
- ✅ Code freeze with strict testing
- ✅ Reproducibility validation
- ✅ Final release preparation

## Test Structure

```
tests/
├── unit/                      # Unit tests for individual functions
│   ├── test_tokenizers.py
│   ├── test_data_utils.py
│   └── test_model_utils.py
├── integration/               # Integration tests
│   ├── test_bert_pipeline.py
│   ├── test_gpt2_pipeline.py
│   └── test_data_loading.py
├── phase1/                    # Phase 1 specific tests
│   ├── test_bert_domains.py
│   └── test_gpt2_domains.py
├── phase2/                    # Phase 2 specific tests
│   ├── test_dataset_preparation.py
│   └── test_training_scripts.py
├── phase3/                    # Phase 3 specific tests
│   ├── test_model_evaluation.py
│   └── test_hyperparameters.py
├── benchmarks/                # Performance benchmarks
│   ├── test_bert_performance.py
│   ├── test_gpt2_performance.py
│   └── test_data_pipeline.py
└── data/                      # Data pipeline tests
    ├── test_genome_loader.py
    ├── test_protein_loader.py
    └── test_compound_loader.py
```

## Required Setup

### 1. Install Development Dependencies
```bash
pip install pytest pytest-cov pytest-xdist pytest-benchmark
pip install mypy types-PyYAML
pip install safety bandit
pip install sphinx sphinx-rtd-theme
```

### 2. Create pytest Configuration
See `pytest.ini` for configuration.

### 3. GitHub Secrets (for later phases)
- `CODECOV_TOKEN`: For code coverage reporting
- `HF_TOKEN`: For Hugging Face model uploads
- `SLACK_WEBHOOK`: For notifications (optional)

## Running Tests Locally

### All tests
```bash
pytest tests/ -v
```

### With coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Specific phase tests
```bash
pytest tests/phase1/ -v
pytest tests/phase2/ -v
pytest tests/phase3/ -v
```

### Benchmarks
```bash
pytest tests/benchmarks/ --benchmark-only
```

## Continuous Integration Workflow

1. **Developer pushes code** → Automatic CI tests run
2. **Pull Request created** → All checks must pass
3. **Code review** → Manual approval required
4. **Merge to develop** → Full test suite runs
5. **Phase completion** → Manual phase validation
6. **Merge to main** → Release workflow triggered

## Best Practices

1. **Write tests alongside features**: For every new feature, add corresponding tests
2. **Keep tests fast**: Unit tests should complete in seconds
3. **Use fixtures**: Share test setup code with pytest fixtures
4. **Mock external dependencies**: Don't rely on external services in CI
5. **Document expected behaviors**: Use clear test names and docstrings
6. **Run tests locally first**: Before pushing, run `pytest` locally
7. **Update phase validation**: When completing tasks, run phase validation workflows

## Monitoring and Reporting

- **Code Coverage**: Tracked via Codecov (once configured)
- **Test Results**: Available in GitHub Actions logs
- **Performance Trends**: Benchmark results stored for comparison
- **Documentation**: Auto-deployed to GitHub Pages on main branch

## Troubleshooting

### Tests failing in CI but passing locally
- Check Python version differences
- Verify all dependencies are specified
- Look for environment-specific issues

### Slow CI runs
- Use pytest-xdist for parallel execution
- Cache dependencies in workflows
- Split large test suites into separate jobs

### Flaky tests
- Identify and fix non-deterministic behavior
- Use `pytest-timeout` to catch hanging tests
- Consider marking as `@pytest.mark.flaky` temporarily

## Next Steps

1. Create initial test files in `tests/` directory
2. Configure pytest with `pytest.ini`
3. Add test coverage to critical paths
4. Set up Codecov integration (optional)
5. Run phase validation workflows for current phase
