# RIKEN Dataset Foundational Model Project

## End-to-End View of the Five-Model Foundation Platform

## Project Overview

An integrated platform to build and evaluate five foundational models across biology and chemistry domains using large-scale datasets.

### Goals

- Multi-modal learning across genome, protein, RNA, compounds, and natural language
- Automated evaluation pipelines
- Web interface for tracking data preparation and model evaluation progress

## Five Model Tracks

### 1. Genome Sequence Model

- Architecture: BERT / GPT-2
- Data source: NCBI RefSeq
- Main tasks: pathogenicity prediction (ClinVar), cancer mutation classification (COSMIC), disease relevance (OMIM)

### 2. Protein Sequence Model

- Architecture: BERT / GPT-2
- Data source: UniProt (Swiss-Prot + TrEMBL)
- Main tasks: fitness prediction (ProteinGym), protein classification

### 3. RNA Expression Model

- Architecture: BERT
- Data source: CELLxGENE (single-cell RNA-seq)
- Main tasks: cell-type classification, expression-pattern analysis

### 4. Molecule Natural Language Model

- Architecture: BERT / GPT-2
- Data source: SMolInstruct (~3.2M molecule-text pairs)
- Main tasks: property prediction from molecular structure, SMILES/NL conversion, instruction-based molecular design

### 5. Compounds Model

- Architecture: BERT / GPT-2
- Data source: OrganiX13 (PubChem, ChEMBL, ZINC)
- Main tasks: compound property prediction, SMILES modeling

## System Architecture

- Web frontend (React.js)
- Backend API (Node.js/Express)
- Python pipelines for:
  - data preparation
  - model training
  - evaluation
  - report generation

## Unified Data Flow

### Phase 1: Data Preparation

1. Download/fetch
2. Format conversion
3. Tokenization
4. Parquet/database conversion

### Phase 2: Model Training

1. Build tokenizer
2. Initialize model
3. Run training loop
4. Save checkpoints

### Phase 3: Evaluation

1. Run evaluation scripts
2. Compute metrics
3. Generate plots
4. Build HTML/PDF reports

## Visualization Outputs

Typical outputs include confusion matrix, ROC, PR curve, metric bars, score histograms, radar charts, scatter plots, and summary dashboards.

## Example Commands

```bash
# Genome data preparation
python src/preparation/preparation_script_genome_sequence.py \
    assets/configs/genome_config.yaml

# Molecule NL preparation
bash src/preparation/download_smolinstruct.sh
python src/preparation/preparation_script_molecule_related_nat_lang.py \
    assets/configs/molecule_nat_lang_config.yaml
```

## Value Proposition

- Unified operations across five modalities
- Reproducible evaluation workflow
- Practical web UI for experiment visibility
- Strong foundation for future model expansion
