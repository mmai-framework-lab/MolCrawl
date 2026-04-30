# Repository File Tree

All non-hidden files tracked by Git, in tree format with brief descriptions.

> Generated: 2026-03-09
> Branch: `fix/custom_tokenizer`

```
riken-dataset-fundational-model/
├── environment.yaml                          # Conda environment definition (Python version + core packages)
├── pyproject.toml                            # Project metadata and tool configuration (Ruff, pytest, etc.)
├── README.md                                 # Top-level project overview and quickstart guide
├── requirements.txt                          # Pip dependency list for non-Conda environments
├── setup.cfg                                 # Legacy setuptools configuration
├── setup.py                                  # Package installation entry point
├── sitecustomize.py                          # Python site-customization hook (import path adjustments)
├── test_image_system.py                      # Ad-hoc test script for the image management system
│
├── assets/
│   ├── logging_config.json                   # JSON configuration for the Python logging system
│   ├── configs/
│   │   ├── compounds.yaml                    # Dataset config for compound (SMILES/scaffold) modality
│   │   ├── genome_sequence.yaml              # Dataset config for genome sequence modality
│   │   ├── molecule_nat_lang_config.yaml     # Dataset config for molecule natural-language modality
│   │   ├── molecules_nl.yaml                 # Alternate config for molecule NL modality
│   │   ├── omim_real_data.template.yaml      # Template config for OMIM real-data evaluation
│   │   ├── protein_sequence.yaml             # Dataset config for protein sequence modality
│   │   └── rna.yaml                          # Dataset config for RNA modality
│   ├── genome_species_list/
│   │   ├── filtered_species_refseq/
│   │   │   ├── bacteria.txt                  # Filtered RefSeq bacteria species list
│   │   │   ├── fungi.txt                     # Filtered RefSeq fungi species list
│   │   │   ├── protozoa.txt                  # Filtered RefSeq protozoa species list
│   │   │   ├── vertebrate_mammalian.txt      # Filtered RefSeq mammalian vertebrate species list
│   │   │   └── vertebrate_other.txt          # Filtered RefSeq other vertebrate species list
│   │   └── species/
│   │       ├── bacteria.txt                  # Full bacteria species list
│   │       ├── fungi.txt                     # Full fungi species list
│   │       ├── invertebrate.txt              # Full invertebrate species list
│   │       ├── protozoa.txt                  # Full protozoa species list
│   │       ├── vertebrate_mammalian.txt      # Full mammalian vertebrate species list
│   │       └── vertebrate_other.txt          # Full other vertebrate species list
│   ├── img/
│   │   ├── compounds_tokenized_Scaffolds_lengths_dist.png   # Token-length distribution chart for compound scaffolds
│   │   ├── compounds_tokenized_SMILES_lengths_dist.png      # Token-length distribution chart for SMILES strings
│   │   ├── genome_sequence_tokenized_lengths_dist.png       # Token-length distribution chart for genome sequences
│   │   ├── molecule_nat_lang_tokenized_test_lengths_dist.png   # Token-length distribution (mol-NL test set)
│   │   ├── molecule_nat_lang_tokenized_train_lengths_dist.png  # Token-length distribution (mol-NL train set)
│   │   ├── molecule_nat_lang_tokenized_validation_lengths_dist.png  # Token-length distribution (mol-NL validation set)
│   │   ├── molecule_nat_lang_tokenized_valid_lengths_dist.png       # Token-length distribution (mol-NL valid split)
│   │   ├── protein_sequence_tokenized_lengths_dist.png      # Token-length distribution chart for protein sequences
│   │   ├── rna_tokenized_lengths_dist.png                   # Token-length distribution chart for RNA sequences
│   │   ├── smiles1.png                                      # Example SMILES structure image 1
│   │   ├── smiles2.png                                      # Example SMILES structure image 2
│   │   └── smiles3.png                                      # Example SMILES structure image 3
│   └── molecules/
│       └── vocab.txt                                        # Vocabulary file for molecule tokenizer
│
├── docs/
│   ├── README.md                             # Documentation index and navigation guide
│   ├── 01-getting_started/
│   │   ├── LEARNING_SOURCE_DIR_STRUCTURE.ja.md   # LEARNING_SOURCE_DIR directory structure (Japanese)
│   │   ├── LEARNING_SOURCE_DIR_STRUCTURE.md      # LEARNING_SOURCE_DIR directory structure (English)
│   │   ├── README_config.ja.md               # Configuration guide (Japanese)
│   │   ├── README_config.md                  # Configuration guide (English)
│   │   ├── README.ja.md                      # Getting started guide (Japanese)
│   │   ├── README.md                         # Getting started guide (English)
│   │   ├── README_webinterface.ja.md         # Web interface guide (Japanese)
│   │   └── README_webinterface.md            # Web interface guide (English)
│   ├── 02-datasets/
│   │   ├── COMPOUNDS_VALIDATION_EXAMPLES.ja.md          # Compound validation examples (Japanese)
│   │   ├── COMPOUNDS_VALIDATION_EXAMPLES.md             # Compound validation examples (English)
│   │   ├── COMPOUNDS_VALIDATION_GUIDE.ja.md             # Compound dataset validation guide (Japanese)
│   │   ├── COMPOUNDS_VALIDATION_GUIDE.md                # Compound dataset validation guide (English)
│   │   ├── COMPOUNDS_VALIDATION_SUMMARY.ja.md           # Compound validation summary (Japanese)
│   │   ├── COMPOUNDS_VALIDATION_SUMMARY.md              # Compound validation summary (English)
│   │   ├── GENOME_SPECIES_API_PERFORMANCE.ja.md         # Genome species API performance report (Japanese)
│   │   ├── GENOME_SPECIES_API_PERFORMANCE.md            # Genome species API performance report (English)
│   │   ├── MOLCRAWL_DATASET_BROWSER_GUIDE.ja.md         # Dataset browser usage guide (Japanese)
│   │   ├── MOLCRAWL_DATASET_BROWSER_GUIDE.md            # Dataset browser usage guide (English)
│   │   ├── MOLCRAWL_DATASET_BROWSER_IMPLEMENTATION_REPORT.ja.md  # Browser implementation report (Japanese)
│   │   ├── MOLCRAWL_DATASET_BROWSER_IMPLEMENTATION_REPORT.md     # Browser implementation report (English)
│   │   ├── ZINC20_DOWNLOAD_AND_CONVERSION_GUIDE.ja.md   # ZINC20 download and conversion guide (Japanese)
│   │   └── ZINC20_DOWNLOAD_AND_CONVERSION_GUIDE.md      # ZINC20 download and conversion guide (English)
│   ├── 03-training/
│   │   ├── README_bert.ja.md                 # BERT training guide (Japanese)
│   │   ├── README_bert.md                    # BERT training guide (English)
│   │   ├── README_gpt2.ja.md                 # GPT-2 training guide (Japanese)
│   │   ├── README_gpt2.md                    # GPT-2 training guide (English)
│   │   ├── README_molecule_nat_lang_training.ja.md  # Molecule NL training guide (Japanese)
│   │   └── README_molecule_nat_lang_training.md     # Molecule NL training guide (English)
│   ├── 04-evaluation/
│   │   ├── README_bert_tester.ja.md          # BERT evaluation guide (Japanese)
│   │   ├── README_bert_tester.md             # BERT evaluation guide (English)
│   │   ├── README_gpt2_tester.ja.md          # GPT-2 evaluation guide (Japanese)
│   │   └── README_gpt2_tester.md             # GPT-2 evaluation guide (English)
│   ├── 05-experiment_tracking/
│   │   ├── EXPERIMENT_TRACKING_ARCHITECTURE.ja.md   # Experiment tracking architecture overview (Japanese)
│   │   ├── EXPERIMENT_TRACKING_ARCHITECTURE.md      # Experiment tracking architecture overview (English)
│   │   ├── EXPERIMENT_TRACKING_QUICKSTART.ja.md     # Experiment tracking quickstart (Japanese)
│   │   ├── EXPERIMENT_TRACKING_QUICKSTART.md        # Experiment tracking quickstart (English)
│   │   ├── EXPERIMENT_TRACKING_README.ja.md         # Experiment tracking module README (Japanese)
│   │   ├── EXPERIMENT_TRACKING_README.md            # Experiment tracking module README (English)
│   │   ├── EXPERIMENT_TRACKING_SUMMARY.ja.md        # Experiment tracking feature summary (Japanese)
│   │   └── EXPERIMENT_TRACKING_SUMMARY.md           # Experiment tracking feature summary (English)
│   ├── 06-operations/
│   │   ├── README_proteingym_bert.ja.md      # ProteinGym BERT evaluation operations (Japanese)
│   │   ├── README_proteingym_bert.md         # ProteinGym BERT evaluation operations (English)
│   │   ├── README_proteingym_gpt2.ja.md      # ProteinGym GPT-2 evaluation operations (Japanese)
│   │   └── README_proteingym_gpt2.md         # ProteinGym GPT-2 evaluation operations (English)
│   ├── 07-reports/
│   │   ├── bert_training_verification_learning_20251125.ja.md  # BERT training verification report (Japanese)
│   │   ├── bert_training_verification_learning_20251125.md     # BERT training verification report (English)
│   │   ├── genome_sequence_compatibility_verification.ja.md    # Genome sequence compatibility report (Japanese)
│   │   ├── genome_sequence_compatibility_verification.md       # Genome sequence compatibility report (English)
│   │   ├── gpt2_training_verification_learning_20251125.ja.md  # GPT-2 training verification report (Japanese)
│   │   ├── gpt2_training_verification_learning_20251125.md     # GPT-2 training verification report (English)
│   │   ├── molecule_nat_lang_dataset_comparison_report.ja.md   # Molecule NL dataset comparison report (Japanese)
│   │   ├── molecule_nat_lang_dataset_comparison_report.md      # Molecule NL dataset comparison report (English)
│   │   ├── presen.ja.md                      # Presentation material (Japanese)
│   │   ├── presen.md                         # Presentation material (English)
│   │   ├── SYSTEM_STARTED.ja.md              # System launch announcement (Japanese)
│   │   ├── SYSTEM_STARTED.md                 # System launch announcement (English)
│   │   ├── verification_in_compounds.ja.md   # Compound dataset verification report (Japanese)
│   │   └── verification_in_compounds.md      # Compound dataset verification report (English)
│   ├── 08-archive/
│   │   ├── 20251104_LEARNING_SOURCE_DIR_MIGRATION.ja.md  # LEARNING_SOURCE_DIR migration notes (Japanese)
│   │   ├── 20251104_LEARNING_SOURCE_DIR_MIGRATION.md     # LEARNING_SOURCE_DIR migration notes (English)
│   │   ├── README_gpt2_train.ja.md           # Archived GPT-2 training notes (Japanese)
│   │   └── README_gpt2_train.md              # Archived GPT-2 training notes (English)
│   ├── 09-future_models/
│   │   ├── CHEMBERTA2_IMPLEMENTATION_SUMMARY.ja.md  # ChemBERTa-2 implementation summary (Japanese)
│   │   ├── CHEMBERTA2_IMPLEMENTATION_SUMMARY.md     # ChemBERTa-2 implementation summary (English)
│   │   ├── CHEMBERTA2_TRAINING_GUIDE.ja.md          # ChemBERTa-2 training guide (Japanese)
│   │   ├── CHEMBERTA2_TRAINING_GUIDE.md             # ChemBERTa-2 training guide (English)
│   │   ├── DNABERT2_IMPLEMENTATION_SUMMARY.ja.md    # DNABERT-2 implementation summary (Japanese)
│   │   ├── DNABERT2_IMPLEMENTATION_SUMMARY.md       # DNABERT-2 implementation summary (English)
│   │   ├── DNABERT2_TRAINING_GUIDE.ja.md            # DNABERT-2 training guide (Japanese)
│   │   ├── DNABERT2_TRAINING_GUIDE.md               # DNABERT-2 training guide (English)
│   │   ├── ESM2_IMPLEMENTATION_SUMMARY.ja.md        # ESM-2 protein model implementation summary (Japanese)
│   │   ├── ESM2_IMPLEMENTATION_SUMMARY.md           # ESM-2 protein model implementation summary (English)
│   │   ├── ESM2_TRAINING_GUIDE.ja.md                # ESM-2 training guide (Japanese)
│   │   ├── ESM2_TRAINING_GUIDE.md                   # ESM-2 training guide (English)
│   │   ├── README_gpn.ja.md                         # GPN (Genomic Pre-trained Network) notes (Japanese)
│   │   ├── README_gpn.md                            # GPN (Genomic Pre-trained Network) notes (English)
│   │   ├── RNAFORMER_IMPLEMENTATION_SUMMARY.ja.md   # RNAformer implementation summary (Japanese)
│   │   ├── RNAFORMER_IMPLEMENTATION_SUMMARY.md      # RNAformer implementation summary (English)
│   │   ├── RNAFORMER_TRAINING_GUIDE.ja.md           # RNAformer training guide (Japanese)
│   │   └── RNAFORMER_TRAINING_GUIDE.md              # RNAformer training guide (English)
│   └── 10-file-tree/
│       └── FILE_TREE.md                      # (this file) Full repository file tree with descriptions
│
├── misc/
│   └── experiment_tracker_sample.py          # Sample script demonstrating experiment tracker usage
│
├── molcrawl/                                 # Main Python package (4-layer layout: core / data / models / tasks)
│   ├── __init__.py                           # Package initializer
│   ├── core/                                 # Horizontal infrastructure (paths, tracking, utilities)
│   │   ├── __init__.py
│   │   ├── base.py                           # Abstract base classes shared across modalities
│   │   ├── config.py                         # Core config dataclasses and validation
│   │   ├── dataset.py                        # Base dataset class for all modalities
│   │   ├── env.sh                            # Shell script to export common environment variables
│   │   ├── paths.py                          # Centralized path constants for the project
│   │   ├── tracking/
│   │   │   ├── __init__.py
│   │   │   ├── api.py                        # REST API interface for the experiment tracker
│   │   │   ├── database.py                   # SQLite database layer for experiment records
│   │   │   ├── helpers.py                    # Utility helpers for experiment tracker
│   │   │   ├── models.py                     # Data models (dataclasses/ORM) for experiments
│   │   │   └── tracker.py                    # Core experiment tracking logic
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── base_visualization.py         # Base class and utilities for result visualization
│   │       ├── cache_config.py               # Configuration caching helpers
│   │       ├── environment_check.py          # Checks that required environment variables are set
│   │       ├── evaluation_output.py          # Handles formatting and saving of evaluation outputs
│   │       ├── get_image_path.py             # Resolves paths for model/dataset image assets
│   │       ├── get_model_images.py           # Retrieves model card images from disk
│   │       ├── image_manager.py              # Manages image storage and retrieval for the web UI
│   │       ├── model_evaluator.py            # Common evaluation loop used across modalities
│   │       └── trainer_utils.py              # Trainer helpers shared across model training entrypoints
│   ├── data/                                 # Per-modality data preparation, tokenizers, and datasets
│   │   ├── __init__.py
│   │   ├── compounds/
│   │   │   ├── __init__.py
│   │   │   ├── download_guacamol.py          # Downloads the GuacaMol compound benchmark dataset
│   │   │   ├── preparation.py                # Master preparation script for compounds modality
│   │   │   ├── dataset/
│   │   │   │   ├── dataset_config.py             # Config dataclass for compound datasets
│   │   │   │   ├── hf_converter.py               # Converts compound data to Hugging Face dataset format
│   │   │   │   ├── multi_loader.py               # Loads multiple compound dataset files in parallel
│   │   │   │   ├── prepare_gpt2.py               # Prepares compound dataset for GPT-2 training
│   │   │   │   ├── prepare_gpt2_organix13.py     # Prepares OrganiX13 dataset specifically for GPT-2
│   │   │   │   ├── processor.py                  # Core compound data processing logic
│   │   │   │   ├── tokenizer.py                  # Compound (SMILES/scaffold) tokenizer
│   │   │   │   └── organix13/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── combine_all.py            # Combines all OrganiX13 sub-datasets into one
│   │   │   │       ├── download.py               # Downloads the OrganiX13 compound dataset
│   │   │   │       ├── opv/
│   │   │   │       │   └── prepare_opv.py        # Prepares OPV (organic photovoltaics) subset
│   │   │   │       └── zinc/
│   │   │   │           ├── download_and_convert_to_parquet.py  # Downloads ZINC20 and converts to Parquet
│   │   │   │           └── zinc_complete/
│   │   │   │               └── filelist.txt      # List of ZINC20 chunk files to download
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── config.py                     # Utility config helpers for compound modality
│   │   │       ├── datasets.py                   # Dataset loading helpers for compounds
│   │   │       ├── general.py                    # General utility functions for compound processing
│   │   │       ├── preprocessing.py              # Compound data preprocessing transformations
│   │   │       └── tokenizer.py                  # Tokenizer utility wrappers for compounds
│   │   ├── genome_sequence/
│   │   │   ├── __init__.py
│   │   │   ├── preparation.py                # Master preparation script for genome sequences
│   │   │   ├── dataset/
│   │   │   │   ├── prepare_gpt2.py               # Prepares genome sequence dataset for GPT-2 training
│   │   │   │   ├── sentence_piece_tokenizer.py   # SentencePiece tokenizer adapter for genome sequences
│   │   │   │   ├── tokenizer.py                  # Custom tokenizer for genome sequences
│   │   │   │   ├── train_tokenizer.py            # Script to train SentencePiece tokenizer on genome data
│   │   │   │   ├── clinvar/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── prepare_clinvar.py        # Prepares ClinVar variant data for genome models
│   │   │   │   └── refseq/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── download_full_refseq.py   # Downloads the full RefSeq assembly collection
│   │   │   │       ├── download_refseq.py        # Downloads selected RefSeq assemblies
│   │   │   │       └── fasta_to_raw.py           # Converts FASTA genome files to raw text format
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       └── config.py                     # Utility config helpers for genome sequence modality
│   │   ├── molecule_nat_lang/
│   │   │   ├── __init__.py
│   │   │   ├── download_smolinstruct.sh      # Shell script to download SMolInstruct dataset
│   │   │   ├── preparation.py                # Master preparation script for molecule NL
│   │   │   ├── dataset/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── download.py                   # Downloads the SMolInstruct molecule NL dataset
│   │   │   │   ├── prepare_gpt2.py               # Prepares molecule NL dataset for GPT-2 training
│   │   │   │   └── prepare_mol_instructions.py   # Prepares Mol-Instructions dataset for molecule NL
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── bert_tokenizer.py             # BERT-compatible tokenizer for molecule NL
│   │   │       ├── config.py                     # Utility config helpers for molecule NL modality
│   │   │       ├── general.py                    # General utility functions for molecule NL
│   │   │       ├── tokenizer.py                  # Custom tokenizer for molecule NL text
│   │   │       └── vocab_guard.py                # Vocabulary consistency guard for molecule NL
│   │   ├── protein_sequence/
│   │   │   ├── __init__.py
│   │   │   ├── launch_data_preparation.sh    # Shell script to launch protein sequence data preparation
│   │   │   ├── preparation.py                # Master preparation script for protein sequences
│   │   │   ├── dataset/
│   │   │   │   ├── download_proteingym.py        # Downloads ProteinGym benchmark data
│   │   │   │   ├── prepare_gpt2.py               # Prepares protein sequence dataset for GPT-2 training
│   │   │   │   ├── prepare_proteingym.py         # Prepares ProteinGym dataset for protein models
│   │   │   │   ├── tokenizer.py                  # Custom tokenizer for protein sequences
│   │   │   │   └── uniprot/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── fasta_to_raw.py           # Converts UniProt FASTA files to raw text format
│   │   │   │       └── uniprot_download.py       # Downloads protein sequences from UniProt
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── bert_tokenizer.py             # BERT-compatible tokenizer for protein sequences
│   │   │       └── configs.py                    # Utility config helpers for protein sequence modality
│   │   └── rna/
│   │       ├── __init__.py
│   │       ├── requirements.txt                  # Additional Python dependencies for RNA modality
│   │       ├── preparation.py                    # Master preparation script for RNA modality
│   │       ├── dataset/
│   │       │   ├── prepare_gpt2.py               # Prepares RNA dataset for GPT-2 training
│   │       │   ├── rna_dataset.py                # RNA dataset class (loading + iteration)
│   │       │   ├── tokenization.py               # RNA-specific tokenization logic
│   │       │   ├── celltype/
│   │       │   │   ├── __init__.py
│   │       │   │   └── prepare_celltype.py       # Prepares cell-type-labeled RNA dataset
│   │       │   ├── cellxgene/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── prepare_cellxgene.py      # Orchestrates CellxGene RNA dataset preparation
│   │       │   │   └── script/
│   │       │   │       ├── __init__.py
│   │       │   │       ├── build_list.py         # Builds file list for CellxGene download
│   │       │   │       ├── conv.py               # Format conversion utilities for CellxGene data
│   │       │   │       ├── download.py           # Downloads CellxGene H5AD files
│   │       │   │       ├── h5ad_to_loom.py       # Converts H5AD files to Loom format
│   │       │   │       └── scgpt_tokenization.py # Tokenizes single-cell data in scGPT style
│   │       │   └── geneformer/
│   │       │       ├── gene_median_dictionary.pkl  # Pre-computed median gene expression dictionary
│   │       │       ├── token_dictionary.pkl         # Gene-to-token mapping for Geneformer tokenizer
│   │       │       └── tokenizer.py                 # Geneformer-style RNA tokenizer
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── bert_tokenizer.py             # BERT-compatible tokenizer for RNA data
│   │           ├── compute_stats.py              # Computes dataset statistics for RNA modality
│   │           ├── config.py                     # Utility config helpers for RNA modality
│   │           └── preprocess.py                 # RNA data preprocessing transformations
│   ├── models/                               # Model architecture implementations (decoder / encoder)
│   │   ├── __init__.py
│   │   ├── bert/
│   │   │   ├── __init__.py
│   │   │   ├── configurator.py               # Builds training configs for BERT models
│   │   │   ├── main.py                       # BERT pre-training entry point
│   │   │   └── test_checkpoint.py            # Verifies a BERT checkpoint loads correctly
│   │   ├── chemberta2/
│   │   │   ├── __init__.py
│   │   │   ├── configurator.py               # Builds training configs for ChemBERTa-2
│   │   │   └── main.py                       # ChemBERTa-2 training entry point
│   │   ├── dnabert2/
│   │   │   ├── __init__.py
│   │   │   ├── configurator.py               # Builds training configs for DNABERT-2
│   │   │   └── main.py                       # DNABERT-2 training entry point
│   │   ├── esm2/
│   │   │   ├── __init__.py
│   │   │   ├── configurator.py               # Builds training configs for ESM-2
│   │   │   └── main.py                       # ESM-2 protein model training entry point
│   │   ├── gpt2/
│   │   │   ├── __init__.py
│   │   │   ├── configurator.py               # Builds training configs for GPT-2 models
│   │   │   ├── model.py                      # GPT-2 model definition and customizations
│   │   │   ├── test_checkpoint.py            # Verifies a GPT-2 checkpoint loads and runs correctly
│   │   │   ├── test_helper.py                # Common helpers shared across GPT-2 test scripts
│   │   │   ├── train.py                      # GPT-2 pre-training main loop
│   │   │   └── test_configs/
│   │   │       ├── __init__.py
│   │   │       ├── compounds_test_config.py          # GPT-2 test config for compounds modality
│   │   │       ├── genome_test_config.py             # GPT-2 test config for genome sequence modality
│   │   │       ├── molecule_nat_lang_test_config.py  # GPT-2 test config for molecule NL modality
│   │   │       ├── protein_sequence_test_config.py   # GPT-2 test config for protein sequence modality
│   │   │       └── rna_test_config.py                # GPT-2 test config for RNA modality
│   │   └── rnaformer/
│   │       ├── __init__.py
│   │       ├── configurator.py               # Builds training configs for RNAformer
│   │       └── main.py                       # RNAformer training entry point
│   ├── preparation/                          # Residual generic data-preparation utilities
│   │   ├── __init__.py
│   │   ├── convert_parquet_to_arrow.py       # Converts Parquet files to Arrow format
│   │   └── test_molecule_nat_lang_compatibility.py   # Tests compatibility of molecule NL processed data
│   ├── tasks/                                # Training and evaluation tasks (organized by purpose)
│   │   ├── __init__.py
│   │   ├── pretrain/
│   │   │   ├── __init__.py
│   │   │   └── configs/                      # Pre-training configs: <modality>/<arch>[_<variant>][_<size>].py
│   │   │       ├── __init__.py
│   │   │       ├── compounds/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── bert_small.py / bert_medium.py / bert_large.py
│   │   │       │   ├── bert_chembl_{small,medium,large}.py
│   │   │       │   ├── bert_guacamol_{small,medium,large}.py
│   │   │       │   ├── chemberta2.py         # Size selected at runtime
│   │   │       │   ├── gpt2_{small,medium,large,xl}.py
│   │   │       │   ├── gpt2_chembl_{small,medium,large,xl}.py
│   │   │       │   └── gpt2_guacamol_{small,medium,large,xl}.py
│   │   │       ├── genome_sequence/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── bert_{small,medium,large}.py
│   │   │       │   ├── bert_clinvar_{small,medium,large}.py
│   │   │       │   ├── dnabert2.py           # Size selected at runtime
│   │   │       │   ├── gpt2_{small,medium,large,xl}.py
│   │   │       │   └── gpt2_clinvar_{small,medium,large,xl}.py
│   │   │       ├── molecule_nat_lang/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── bert_{small,medium,large}.py
│   │   │       │   ├── bert_mol_instructions_{small,medium,large}.py
│   │   │       │   ├── gpt2_{small,medium,large,xl}.py
│   │   │       │   └── gpt2_mol_instructions_{small,medium,large,xl}.py
│   │   │       ├── protein_sequence/
│   │   │       │   ├── __init__.py
│   │   │       │   ├── bert_{small,medium,large}.py
│   │   │       │   ├── bert_proteingym_{small,medium,large}.py
│   │   │       │   ├── esm2.py               # Size selected at runtime
│   │   │       │   ├── gpt2_{small,medium,large,xl}.py
│   │   │       │   └── gpt2_proteingym_{small,medium,large,xl}.py
│   │   │       └── rna/
│   │   │           ├── __init__.py
│   │   │           ├── bert_{small,medium,large}.py
│   │   │           ├── bert_celltype_{small,medium,large}.py
│   │   │           ├── gpt2_{small,medium,large,xl}.py
│   │   │           ├── gpt2_celltype_{small,medium,large,xl}.py
│   │   │           └── rnaformer.py          # Size selected at runtime
│   │   ├── evaluation/                       # Benchmarks organized by task (arch as file prefix)
│   │   │   ├── __init__.py
│   │   │   ├── clinvar/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bert_config.py            # BERT ClinVar evaluation config
│   │   │   │   ├── bert_evaluation.py        # BERT ClinVar evaluation
│   │   │   │   ├── bert_visualization.py     # BERT ClinVar result visualization
│   │   │   │   ├── extract_random_samples.py # Random-sample extraction helper
│   │   │   │   ├── gpt2_data_preparation.py  # GPT-2 ClinVar data preparation
│   │   │   │   ├── gpt2_evaluation.py        # GPT-2 ClinVar evaluation
│   │   │   │   ├── gpt2_visualization.py     # GPT-2 ClinVar result visualization
│   │   │   │   └── prepare_sequences.py      # Extracts sequences from raw ClinVar VCF
│   │   │   ├── cosmic/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── gpt2_data_preparation.py  # GPT-2 COSMIC data preparation
│   │   │   │   ├── gpt2_evaluation.py        # GPT-2 COSMIC evaluation
│   │   │   │   └── gpt2_visualization.py     # GPT-2 COSMIC result visualization
│   │   │   ├── molecule_nat_lang/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bert_evaluation.py        # BERT molecule-NL evaluation
│   │   │   │   ├── gpt2_evaluation.py        # GPT-2 molecule-NL evaluation
│   │   │   │   └── gpt2_visualization.py     # GPT-2 molecule-NL visualization
│   │   │   ├── omim/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── gpt2_data_preparation.py  # GPT-2 OMIM data preparation
│   │   │   │   ├── gpt2_evaluation.py        # GPT-2 OMIM evaluation
│   │   │   │   ├── gpt2_real_data_processor.py  # Data processor for real OMIM data
│   │   │   │   └── gpt2_visualization.py     # GPT-2 OMIM visualization
│   │   │   ├── protein_classification/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── gpt2_data_preparation.py
│   │   │   │   ├── gpt2_evaluation.py
│   │   │   │   └── gpt2_visualization.py
│   │   │   ├── proteingym/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bert_config.py            # BERT ProteinGym evaluation config
│   │   │   │   ├── bert_data_preparation.py
│   │   │   │   ├── bert_evaluation.py
│   │   │   │   ├── bert_visualization.py
│   │   │   │   ├── gpt2_data_preparation.py
│   │   │   │   ├── gpt2_evaluation.py
│   │   │   │   └── gpt2_visualization.py
│   │   │   └── rna_benchmark/
│   │   │       ├── __init__.py
│   │   │       ├── data_preparation.py       # RNA benchmark data preparation
│   │   │       └── evaluation.py             # RNA benchmark evaluation
│   │   └── downstream/                       # Multimodal downstream tasks
│   │       ├── __init__.py
│   │       └── compound_protein/
│   │           └── __init__.py               # Skeleton for compound-protein dual-encoder and conditional generator

│
├── molcrawl-web/                             # Web-based dataset browser (React + Express)
│   ├── package.json                          # npm dependencies and scripts for the monorepo root
│   ├── package-lock.json                     # Locked npm dependency tree
│   ├── server.js                             # Express backend API server entry point
│   ├── check-config.js                       # Script to validate environment configuration
│   ├── get_learning_source_dir.py            # Python helper to resolve LEARNING_SOURCE_DIR
│   ├── README.md                             # Web interface documentation and usage guide
│   ├── ESLINT_SETUP.md                       # ESLint configuration documentation
│   ├── INFERENCE_FEATURE.md                  # Documentation for the model inference feature
│   ├── INFERENCE_IMPLEMENTATION_SUMMARY.md   # Implementation summary for inference feature
│   ├── INFERENCE_VISUAL_GUIDE.md             # Visual walkthrough of the inference UI
│   ├── TROUBLESHOOTING.md                    # Common issues and solutions for the web UI
│   ├── start-both.sh                         # Starts both frontend and backend in separate processes
│   ├── start-dev.sh                          # Development start script with NFS auto-detection
│   ├── start-new.sh                          # Alternative start script (newer variant)
│   ├── start.sh                              # Simple start script
│   ├── api/
│   │   ├── bert-inference.js                 # API handler for BERT model inference requests
│   │   ├── bert-training-status.js           # API handler for BERT training status queries
│   │   ├── dataset-progress.js               # API handler for dataset preparation progress
│   │   ├── directory.js                      # API handler for filesystem directory browsing
│   │   ├── genome-species.js                 # API handler for genome species list queries
│   │   ├── gpt2-inference.js                 # API handler for GPT-2 model inference requests
│   │   ├── gpt2-training-status.js           # API handler for GPT-2 training status queries
│   │   ├── gpu-resources.js                  # API handler for GPU resource monitoring
│   │   ├── images.js                         # API handler for model/dataset image serving
│   │   ├── logs.js                           # API handler for log file streaming
│   │   ├── preparation-runner.js             # API handler for launching preparation scripts
│   │   ├── training-process-status.js        # API handler for training process status
│   │   ├── wandb-experiments.js              # API handler for W&B experiment data
│   │   └── zinc-checker.js                   # API handler for ZINC20 data availability checks
│   ├── frontend/                             # Legacy/alternate frontend (CRA project)
│   │   ├── package.json                      # npm config for legacy frontend
│   │   ├── public/
│   │   │   ├── index.html                    # HTML entry point for legacy frontend
│   │   │   └── manifest.json                 # PWA manifest for legacy frontend
│   │   └── src/
│   │       ├── App.css                       # Styles for legacy App component
│   │       ├── App.js                        # Legacy React application root
│   │       ├── components/
│   │       │   ├── DatasetInfo.css           # Styles for legacy DatasetInfo component
│   │       │   ├── DatasetInfo.js            # Legacy DatasetInfo React component
│   │       │   ├── DirectoryViewer.css       # Styles for legacy DirectoryViewer component
│   │       │   ├── DirectoryViewer.js        # Legacy DirectoryViewer React component
│   │       │   ├── ImageGallery.css          # Styles for legacy ImageGallery component
│   │       │   └── ImageGallery.js           # Legacy ImageGallery React component
│   │       ├── index.css                     # Global styles for legacy frontend
│   │       └── index.js                      # Legacy React entry point
│   ├── public/
│   │   ├── disable-hmr.js                    # Disables Hot Module Replacement (for NFS stability)
│   │   ├── favicon.ico                       # Browser tab icon
│   │   ├── index.html                        # HTML entry point for the main frontend
│   │   ├── logo192.png                       # PWA icon (192×192)
│   │   └── logo512.png                       # PWA icon (512×512)
│   └── src/
│       ├── App.css                           # Global styles for the main App component
│       ├── App.js                            # Main React application root and routing
│       ├── App.test.js                       # Unit tests for the App component
│       ├── BERTInferenceModal.css            # Styles for BERT inference modal dialog
│       ├── BERTInferenceModal.js             # BERT inference modal React component
│       ├── BERTTrainingStatus.css            # Styles for BERT training status panel
│       ├── BERTTrainingStatus.js             # BERT training status React component
│       ├── DatasetProgressCard.css           # Styles for dataset progress card component
│       ├── DatasetProgressCard.js            # Dataset preparation progress card component
│       ├── ExperimentDashboard.css           # Styles for experiment dashboard component
│       ├── ExperimentDashboard.js            # Experiment tracking dashboard React component
│       ├── GenomeSpeciesList.css             # Styles for genome species list component
│       ├── GenomeSpeciesList.js              # Genome species list React component
│       ├── GPT2TrainingStatus.css            # Styles for GPT-2 training status panel
│       ├── GPT2TrainingStatus.js             # GPT-2 training status React component
│       ├── GPUResources.css                  # Styles for GPU resources monitor
│       ├── GPUResources.js                   # GPU resource monitoring React component
│       ├── InferenceModal.css                # Styles for generic inference modal dialog
│       ├── InferenceModal.js                 # Generic model inference modal React component
│       ├── LogsViewer.css                    # Styles for logs viewer component
│       ├── LogsViewer.js                     # Training/preparation log streaming component
│       ├── TrainingProcessStatus.css         # Styles for training process status component
│       ├── TrainingProcessStatus.js          # Overall training process status component
│       ├── ZincChecker.css                   # Styles for ZINC20 checker component
│       ├── ZincChecker.js                    # ZINC20 data availability checker component
│       ├── index.css                         # Global CSS reset and base styles
│       ├── index.js                          # React DOM entry point
│       ├── logo.svg                          # MolCrawl SVG logo
│       ├── reportWebVitals.js                # Web Vitals performance reporting helper
│       ├── setupProxy.js                     # CRA proxy configuration (dev API proxying)
│       ├── setupTests.js                     # Jest test setup (testing-library config)
│       ├── components/
│       │   ├── LanguageSwitcher.css          # Styles for language switcher component
│       │   └── LanguageSwitcher.js           # EN/JA language toggle React component
│       └── i18n/
│           ├── I18nContext.js                # React context provider for internationalization
│           ├── index.js                      # i18n module entry point
│           └── locales/
│               ├── en.json                   # English UI string translations
│               └── ja.json                   # Japanese UI string translations
│
├── tests/
│   ├── README.md                             # Test suite overview and instructions
│   ├── CI_QUICKSTART.md                      # CI quick-start guide
│   ├── CI_SETUP_REPORT.md                    # CI setup report
│   ├── PHASE_PROGRESS.md                     # Test phase progress tracking
│   ├── conftest.py                           # Pytest fixtures shared across all tests
│   ├── integration/
│   │   ├── test_bert_pipeline.py             # Integration test for the full BERT pipeline
│   │   ├── test_compounds_pipeline.py        # Integration test for the compounds pipeline
│   │   └── test_gpt2_pipeline.py             # Integration test for the full GPT-2 pipeline
│   ├── phase1/
│   │   ├── test_bert_domains.py              # Phase 1 tests: BERT across all modality domains
│   │   └── test_gpt2_domains.py              # Phase 1 tests: GPT-2 across all modality domains
│   ├── phase2/
│   │   └── test_dataset_preparation.py       # Phase 2 tests: dataset preparation correctness
│   ├── phase3/
│   │   └── test_model_evaluation.py          # Phase 3 tests: model evaluation metrics
│   └── unit/
│       ├── test_compounds.py                 # Unit tests for compound processing utilities
│       ├── test_data_utils.py                # Unit tests for data utility functions
│       └── test_tokenizers.py                # Unit tests for all modality tokenizers
│
└── workflows/
    ├── README.md                             # Workflow scripts overview and usage instructions
    ├── common_functions.sh                   # Shared shell functions sourced by workflow scripts
    ├── 00-first.sh                           # One-time initial setup script
    ├── 01-compounds_guacamol-prepare.sh      # Phase 1: prepare GuacaMol compound dataset
    ├── 01-compounds_prepare.sh               # Phase 1: prepare OrganiX13 compound dataset
    ├── 01-genome_sequence-prepare.sh         # Phase 1: prepare RefSeq genome sequence dataset
    ├── 01-molecule_nat_lang-prepare.sh       # Phase 1: prepare SMolInstruct molecule NL dataset
    ├── 01-protein_sequence-prepare.sh        # Phase 1: prepare UniProt protein sequence dataset
    ├── 01-rna-prepare.sh                     # Phase 1: prepare CellxGene RNA dataset
    ├── 02-compounds_organix13-prepare-gpt2.sh  # Phase 2: tokenize OrganiX13 data for GPT-2
    ├── 02-compounds-prepare-gpt2.sh          # Phase 2: tokenize compound data for GPT-2
    ├── 02-genome_sequence-prepare-gpt2.sh    # Phase 2: tokenize genome data for GPT-2
    ├── 02-molecule_nat_lang-prepare-gpt2.sh  # Phase 2: tokenize molecule NL data for GPT-2
    ├── 02-protein_sequence-prepare-gpt2.sh   # Phase 2: tokenize protein data for GPT-2
    ├── 02-rna-prepare-gpt2.sh               # Phase 2: tokenize RNA data for GPT-2
    ├── 03a-compounds_guacamol-train-large.sh   # GPT-2 Large training: GuacaMol compounds
    ├── 03a-compounds_guacamol-train-medium.sh  # GPT-2 Medium training: GuacaMol compounds
    ├── 03a-compounds_guacamol-train-small.sh   # GPT-2 Small training: GuacaMol compounds
    ├── 03a-compounds_guacamol-train-xl.sh      # GPT-2 XL training: GuacaMol compounds
    ├── 03a-genome_sequence-train-large.sh      # GPT-2 Large training: genome sequences
    ├── 03a-genome_sequence-train-medium.sh     # GPT-2 Medium training: genome sequences
    ├── 03a-genome_sequence-train-small.sh      # GPT-2 Small training: genome sequences
    ├── 03a-genome_sequence-train-xl.sh         # GPT-2 XL training: genome sequences
    ├── 03a-molecule_nat_lang-train-large.sh    # GPT-2 Large training: molecule NL
    ├── 03a-molecule_nat_lang-train-medium.sh   # GPT-2 Medium training: molecule NL
    ├── 03a-molecule_nat_lang-train-small.sh    # GPT-2 Small training: molecule NL
    ├── 03a-molecule_nat_lang-train-xl.sh       # GPT-2 XL training: molecule NL
    ├── 03a-protein_sequence-train-large.sh     # GPT-2 Large training: protein sequences
    ├── 03a-protein_sequence-train-medium.sh    # GPT-2 Medium training: protein sequences
    ├── 03a-protein_sequence-train-small.sh     # GPT-2 Small training: protein sequences
    ├── 03a-protein_sequence-train-xl.sh        # GPT-2 XL training: protein sequences
    ├── 03a-rna-train-large.sh                  # GPT-2 Large training: RNA
    ├── 03a-rna-train-medium.sh                 # GPT-2 Medium training: RNA
    ├── 03a-rna-train-small.sh                  # GPT-2 Small training: RNA
    ├── 03a-rna-train-xl.sh                     # GPT-2 XL training: RNA
    ├── 03b-genome_sequence-train-wandb-small.sh  # GPT-2 Small training: genome with W&B logging
    ├── 03b-rna-train-yigarashi_refined-small.sh  # GPT-2 Small training: RNA (Yigarashi refined config)
    ├── 03c-compounds-train-bert-small.sh        # BERT Small training: compounds
    ├── 03c-genome_sequence-train-bert-small.sh  # BERT Small training: genome sequences
    ├── 03c-molecule_nat_lang-train-bert-small.sh  # BERT Small training: molecule NL
    ├── 03c-protein_sequence-train-bert-small.sh   # BERT Small training: protein sequences
    ├── 03c-rna-train-bert-small.sh              # BERT Small training: RNA
    ├── 03d-genome_sequence-train-dnabert2-large.sh   # DNABERT-2 Large training: genome sequences
    ├── 03d-genome_sequence-train-dnabert2-medium.sh  # DNABERT-2 Medium training: genome sequences
    ├── 03d-genome_sequence-train-dnabert2-small.sh   # DNABERT-2 Small training: genome sequences
    ├── 03e-protein_sequence-train-esm2-large.sh  # ESM-2 Large training: protein sequences
    ├── 03e-protein_sequence-train-esm2-medium.sh # ESM-2 Medium training: protein sequences
    ├── 03e-protein_sequence-train-esm2-small.sh  # ESM-2 Small training: protein sequences
    ├── 03f-rna-train-rnaformer-large.sh          # RNAformer Large training: RNA
    ├── 03f-rna-train-rnaformer-medium.sh         # RNAformer Medium training: RNA
    ├── 03f-rna-train-rnaformer-small.sh          # RNAformer Small training: RNA
    ├── 03g-compounds-train-chemberta2-large.sh   # ChemBERTa-2 Large training: compounds
    ├── 03g-compounds-train-chemberta2-medium.sh  # ChemBERTa-2 Medium training: compounds
    ├── 03g-compounds-train-chemberta2-small.sh   # ChemBERTa-2 Small training: compounds
    ├── batch_test_gpt2.sh                        # Batch test script for multiple GPT-2 checkpoints
    ├── convert_molecule_nat_lang_to_arrow.sh     # Converts molecule NL dataset to Arrow format
    ├── create_sample_vocab.sh                    # Creates a sample vocabulary file for testing
    ├── debug_protein_bert.sh                     # Debug helper for protein sequence BERT issues
    ├── demo_experiment_system.sh                 # Demonstrates the experiment tracking system
    ├── gpt2_test_checkpoint.sh                   # Tests a specific GPT-2 checkpoint via CLI
    ├── reboot-cause-check.sh                     # Checks system logs for unexpected reboot causes
    ├── run_bert_clinvar_evaluation.sh            # Runs BERT evaluation on ClinVar variant dataset
    ├── run_bert_proteingym_evaluation.sh         # Runs BERT evaluation on ProteinGym benchmark
    ├── run_gpt2_clinvar_evaluation.sh            # Runs GPT-2 evaluation on ClinVar variant dataset
    ├── run_gpt2_cosmic_evaluation.sh             # Runs GPT-2 evaluation on COSMIC mutation dataset
    ├── run_gpt2_omim_evaluation_dummy.sh         # Runs GPT-2 OMIM evaluation with dummy data
    ├── run_gpt2_omim_evaluation_real.sh          # Runs GPT-2 OMIM evaluation with real data
    ├── run_gpt2_protein_classification.sh        # Runs GPT-2 protein function classification task
    ├── run_gpt2_proteingym_evaluation.sh         # Runs GPT-2 evaluation on ProteinGym benchmark
    ├── run_rna_benchmark_evaluation.sh           # Runs model evaluation on RNA benchmark dataset
    ├── setup_experiment_system.sh                # Sets up the experiment tracking system
    ├── start_api_server.py                       # Python script to start the experiment tracker API
    ├── start_experiment_system.sh                # Starts the full experiment tracking system
    ├── test_huggingface_download.py              # Tests downloading models/datasets from Hugging Face
    ├── test_huggingface_download.sh              # Shell wrapper for HuggingFace download test
    ├── train_rna_yigarashi.sh                    # Training script for RNA model (Yigarashi config)
    ├── upload_to_huggingface.py                  # Python script to upload models to HuggingFace Hub
    ├── upload_to_huggingface.sh                  # Shell wrapper for HuggingFace upload script
    └── web.sh                                    # Starts the MolCrawl web interface
```
