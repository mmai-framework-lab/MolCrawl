# Compounds Validation - Practical Examples

## Real-world usage: verify Compounds processing correctness with GitHub Actions

This document provides concrete, step-by-step examples for validating Compounds processing using the CI/CD system.

##  Scenario 1: You added or modified SMILES validation

### Situation (Scenario 1)

You want to improve the SMILES validation logic in `molcrawl/data/compounds/utils/preprocessing.py`.

### Steps (Scenario 1)

#### 1. Run tests locally

```bash
# Run Compounds-related tests
pytest tests/unit/test_compounds.py::TestSmilesValidation -v

# Expected output:
# tests/unit/test_compounds.py::TestSmilesValidation::test_valid_smiles PASSED
# tests/unit/test_compounds.py::TestSmilesValidation::test_invalid_smiles PASSED
# tests/unit/test_compounds.py::TestSmilesValidation::test_complex_valid_smiles PASSED
```

#### 2. Update the code

```python
# molcrawl/data/compounds/utils/preprocessing.py
def prepare_scaffolds(smiles: str):
    # Add improved logic
    if not smiles or smiles.strip() == "":
        return ""

    # New validation logic
    if len(smiles) > 1000:  # Reject abnormally long SMILES
        logger.warning(f"SMILES too long: {len(smiles)} characters")
        return ""

    # Existing logic...
```

#### 3. Re-run tests locally

```bash
pytest tests/unit/test_compounds.py::TestSmilesValidation -v
```

#### 4. Push to GitHub

```bash
git add molcrawl/data/compounds/utils/preprocessing.py
git commit -m "feat(compounds): add length validation for SMILES"
git push origin feature/improve-smiles-validation
```

#### 5. Check GitHub Actions results

```text
GitHub -> Actions tab -> "Compounds Validation" workflow

 unit-tests: SUCCESS
 smiles-validation: SUCCESS
   Invalid SMILES: 2/100 (2.00%)
   ✓ Invalid SMILES rate is acceptable
```

##  Scenario 2: Add new tokenizer logic

### Situation (Scenario 2)

You want to extend the tokenizer to support special chemical structures (for example stereochemistry).

### Steps (Scenario 2)

#### 1. Develop test-first

```python
# Add to tests/unit/test_compounds.py
@pytest.mark.unit
@pytest.mark.compound
def test_stereochemistry_tokenization(self, sample_vocab_file):
    """Verify stereochemical notation is tokenized correctly"""
    from molcrawl.data.compounds.utils.tokenizer import SmilesTokenizer

    # SMILES including stereochemistry such as `C[C@H](O)C`
    smiles = "C[C@H](O)C"

    tokenizer = SmilesTokenizer(sample_vocab_file)
    tokens = tokenizer.tokenize(smiles)

    # Ensure @ is recognized correctly
    assert "@" in tokens or "[C@H]" in tokens
```

#### 2. Run the test (it should fail)

```bash
pytest tests/unit/test_compounds.py::test_stereochemistry_tokenization -v
# FAILED - expected because the feature is not implemented yet
```

#### 3. Implement the feature

```python
# molcrawl/data/compounds/utils/tokenizer.py
SMI_REGEX_PATTERN = r"""(
    \[[^\]]+]|        # Inside square brackets
    Br?|Cl?|N|O|S|P|F|I|  # Atoms
    b|c|n|o|s|p|      # Aromatic atoms
    @|@@|             # Stereochemistry <- added
    \(|\)|            # Parentheses
    \.|=|#|-|\+|\\|\/|:|~|\?|>>?|\*|\$|
    \%[0-9]{2}|
    [0-9]
)"""
```

#### 4. Re-run the test (it should pass)

```bash
pytest tests/unit/test_compounds.py::test_stereochemistry_tokenization -v
# PASSED
```

#### 5. Run the full test file

```bash
pytest tests/unit/test_compounds.py -v
```

#### 6. Validate on GitHub

```bash
git add molcrawl/data/compounds/utils/tokenizer.py tests/unit/test_compounds.py
git commit -m "feat(compounds): support stereochemistry in tokenizer"
git push

# GitHub Actions runs automatically:
#  unit-tests
#  tokenization-tests
```

##  Scenario 3: Phase 1 BERT model verification

### Situation (Scenario 3)

You trained a compounds BERT model and want to run Phase 1 verification.

### Steps (Scenario 3)

#### 1. Set the model path

```bash
# For local tests
export COMPOUNDS_BERT_MODEL_PATH=/path/to/trained/bert/model
```

#### 2. Run integration tests

```bash
pytest tests/integration/test_compounds_pipeline.py::TestCompoundsBERTIntegration -v

# Expected output:
# test_bert_model_loading PASSED
#   ✓ BERT model loaded successfully
# test_bert_tokenizer_loading PASSED
#   ✓ Tokenizer loaded successfully
#   Vocab size: 500
# test_bert_inference_pipeline PASSED
#   ✓ BERT inference successful
#   Input SMILES: CCO
#   Output shape: torch.Size([1, 10, 500])
```

#### 3. Manually run the Phase 1 verification workflow

```bash
gh workflow run compounds-validation.yml
```

Or from the GitHub Web UI:

```text
Actions -> Compounds Validation -> Run workflow
```

#### 4. Check results

```text
Artifacts:
- Download phase1-compounds-verification-report.md

Contents:
# Compounds Phase 1 Verification Report
Date: 2026-01-05

## BERT Verification Status
-  Model initialization
-  Tokenization pipeline
-  Inference test

Status: PASSED
```

##  Scenario 4: Verify GPT-2 SMILES generation quality

### Situation (Scenario 4)

You want to evaluate the quality of SMILES generated by a GPT-2 model.

### Steps (Scenario 4)

#### 1. Run integration tests

```bash
export COMPOUNDS_GPT2_MODEL_PATH=/path/to/trained/gpt2/model

pytest tests/integration/test_compounds_pipeline.py::TestCompoundsGPT2Integration::test_gpt2_generated_smiles_validity -v
```

#### 2. Check results

```text
✓ SMILES Validity Check:
  Total generated: 15
  Valid SMILES: 12
  Validity rate: 80.0%

PASSED
```

#### 3. Debug when quality is low

```bash
# Run a more detailed test
pytest tests/integration/test_compounds_pipeline.py::TestCompoundsGPT2Integration -v --tb=long

# Inspect generated SMILES:
# 1. CCO
# 2. c1ccccc1O
# 3. INVALID_STRUCTURE <- this is the issue
```

#### 4. Retrain the model or tune hyperparameters

```bash
# Lower temperature (more conservative generation)
# temperature: 0.8 -> 0.6

# Re-test
pytest tests/integration/test_compounds_pipeline.py::TestCompoundsGPT2Integration -v
```

##  Scenario 5: Automatic validation in Pull Requests

### Situation (Scenario 5)

A team member created a PR that changes Compounds-related code.

### Steps (Scenario 5)

#### 1. Create a PR

```bash
# Create a feature branch
git checkout -b feature/add-new-smiles-feature
git add molcrawl/data/compounds/
git commit -m "feat: add new SMILES feature"
git push origin feature/add-new-smiles-feature

# Create a PR on GitHub
```

#### 2. Automatic checks start

```text
GitHub PR page:
 Compounds Validation - unit-tests
 Compounds Validation - integration-tests
 Compounds Validation - smiles-validation
 Compounds Validation - tokenization-tests
 Compounds Validation - phase1-verification (running)
```

#### 3. Reviewer checks results

```text
PR Check Details:

 All checks passed

Details:
- Unit Tests: 25/25 passed
- Integration Tests: 8/8 passed
- Invalid SMILES rate: 3.2% (acceptable)
- Tokenization: All tests passed

Artifacts:
 compounds-validation-summary.md
```

#### 4. Merge

```text
When all checks are , the "Merge pull request" button becomes enabled
```

##  Actual validation flow diagram

```text
Developer changes code
    ↓
Run pytest locally
    ↓
No issues?
    ↓ Yes
Git push
    ↓
GitHub Actions runs automatically
    ├── Unit Tests (1-2 min)
    ├── Integration Tests (3-5 min)
    ├── SMILES Validation (30 sec)
    ├── Tokenization Tests (30 sec)
    └── Phase 1 Verification (optional)
    ↓
All green ?
    ↓ Yes
Generate and store reports
    ↓
PR can be merged
```

##  Common usage patterns

### 1. Daily routine development

```bash
# Morning: pull latest
git pull origin develop

# Add features
vim molcrawl/data/compounds/utils/preprocessing.py

# Add tests
vim tests/unit/test_compounds.py

# Validate locally
pytest tests/unit/test_compounds.py -v

# Push (CI runs automatically)
git push
```

### 2. Weekly model verification

```bash
# Run integration tests weekly (for example every Friday)
gh workflow run compounds-validation.yml

# Review results in the team meeting
```

### 3. Full verification before release

```bash
# At Phase 1 completion
gh workflow run phase-validation.yml -f phase=phase1-bert-verification
gh workflow run phase-validation.yml -f phase=phase1-gpt2-verification

# Run all Compounds tests
gh workflow run compounds-validation.yml -f test_level=all
```

##  How to read metrics

### Example pass criteria

```yaml
Unit Tests:  100% pass required
SMILES Validation:  Invalid rate < 50%
GPT-2 Validity:  > 50% valid SMILES
Test Coverage:  > 80% (target)
Test Duration:  < 5 minutes (target)
```

### When warnings appear

```text
 Invalid SMILES rate: 45%
-> Still acceptable, but investigate the cause

 GPT-2 validity: 48%
-> Slightly below threshold (50%). Consider model tuning

 Unit test failed: 2/25
-> Requires immediate fix
```

##  Summary

What these practical examples show:

1. **Fast feedback**: quickly detect issues through local -> CI -> report flow
2. **Clear quality criteria**: concrete metrics such as invalid SMILES rate and validity rate
3. **Value of automation**: automatic validation on push plus optional manual deep checks
4. **Team collaboration**: automatic PR checks support code review quality

**Next step**: See [COMPOUNDS_VALIDATION_GUIDE.md](COMPOUNDS_VALIDATION_GUIDE.md) for detailed guidance
