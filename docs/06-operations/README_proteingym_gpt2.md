# ProteinGym Evaluation for Protein Sequence GPT-2 Models

This directory contains scripts to evaluate protein-sequence GPT-2 models on the ProteinGym benchmark.

## ProteinGym Dataset

ProteinGym is a large benchmark dataset for variant effect prediction and protein design ([paper](https://pubmed.ncbi.nlm.nih.gov/38106144/)).
In this project, trained GPT-2 protein-sequence models are evaluated against Deep Mutational Scanning (DMS) assay data.

### Available datasets

**Recommended datasets (primary for protein-sequence evaluation):**

- **DMS Substitutions**: Single amino-acid substitution data (main benchmark)
- **DMS Reference**: Assay metadata and references
- **Clinical Substitutions**: Clinically relevant variant substitutions (supplemental)
- **Clinical Reference**: Clinical variant metadata

**Additional datasets:**

- **DMS Indels**: Insertion/deletion variants
- **MSA Files**: Multiple sequence alignments (advanced analysis)
- **Protein Structures**: AlphaFold2 structures (structure-aware analysis)

Data source: [ProteinGym download page](https://proteingym.org/download) (v1.3).

## File Structure

### Main scripts

- `scripts/proteingym_evaluation.py` - Main evaluation script
- `scripts/proteingym_data_preparation.py` - Dataset download and preprocessing
- `scripts/proteingym_visualization.py` - Result visualization
- `run_proteingym_evaluation.sh` - End-to-end runner

### Documentation

- `README_proteingym.md` - This document

## Environment Requirements

### Python packages

```bash
pip install torch numpy pandas scikit-learn sentencepiece scipy matplotlib seaborn requests tqdm biopython
```

### System requirements

- Python 3.8+
- CUDA-capable GPU (recommended; CPU is also possible)
- Sufficient memory depending on assay/data size

## Usage

### 1. Basic evaluation

```bash
# Evaluate a trained model on one ProteinGym assay file
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --data_path proteingym_data/YOUR_ASSAY.csv
```

### 2. Quick test with sample data

```bash
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --create-sample \
    --data_path sample_data.csv \
    --visualize
```

### 3. Auto-download and evaluate

```bash
./run_proteingym_evaluation.sh \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --download-data \
    --visualize
```

### 4. Run individual scripts

#### Data preparation

```bash
# Download recommended datasets
python scripts/proteingym_data_preparation.py --download recommended --data_dir proteingym_data/

# List available assays
python scripts/proteingym_data_preparation.py --list_assays --data_type substitutions

# Get a small set of test assays
python scripts/proteingym_data_preparation.py --get_test_assays 5

# Prepare a specific assay
python scripts/proteingym_data_preparation.py --prepare_assay ASSAY_ID --max_variants 1000

# Download individual dataset categories
python scripts/proteingym_data_preparation.py --download substitutions --data_dir proteingym_data/
python scripts/proteingym_data_preparation.py --download clinical_substitutions --data_dir proteingym_data/
```

#### Evaluation

```bash
python scripts/proteingym_evaluation.py \
    --model_path gpt2-output/protein_sequence-small/ckpt.pt \
    --proteingym_data proteingym_data/ASSAY_ID.csv \
    --output_dir results/ \
    --batch_size 32 \
    --device cuda
```

#### Visualization

```bash
python scripts/proteingym_visualization.py \
    --results_json results/proteingym_results.json \
    --output_dir results/plots
```

## Input Format

ProteinGym CSV input should contain at least:

- `mutant` or mutation descriptor
- `mutated_sequence` (or sequence field used by the script)
- `DMS_score` (target value)

## Main Outputs

- Evaluation JSON summary
- Detailed prediction CSV
- Plots (correlation/scatter/ranking diagnostics)
- Text report

## Metrics

- Spearman correlation
- Pearson correlation
- MAE
- RMSE
- Optional ranking-oriented diagnostics

## Notes

- Use smaller batch sizes when GPU memory is limited.
- For reproducibility, keep model checkpoint, assay ID, and script options logged together.
- Start from the `small` checkpoint to validate your environment before scaling.
