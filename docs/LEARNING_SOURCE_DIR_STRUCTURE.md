# LEARNING_SOURCE_DIR Directory Structure

The `LEARNING_SOURCE_DIR` environment variable points to the main data directory containing training data, model outputs, and logs for different biological sequence types.

```text
LEARNING_SOURCE_DIR/
├── compounds/                    # Chemical compound data
│   ├── benchmark/               # Benchmark datasets
│   │   └── GuacaMol/           # GuacaMol benchmark
│   ├── bert-output/            # BERT model training outputs
│   ├── chemberta2-output/      # ChemBERTA2 model outputs
│   ├── compounds_logs/         # Compound processing logs
│   ├── data/                   # Raw compound datasets
│   │   ├── Fraunhofer-SCAI-llamol/
│   │   ├── opv/
│   │   └── zinc20/
│   ├── gpt2-output/            # GPT-2 model training outputs
│   ├── logs/                   # General logs
│   ├── organix13/              # OrganiX13 dataset
│   │   ├── OrganiX13.parquet
│   │   ├── OrganiX13_tokenized.parquet
│   │   └── compounds/
│   └── zinc_processed.parquet
│
├── genome_sequence/             # Genome sequence data
│   ├── bert-output/            # BERT model outputs
│   ├── data/                   # Raw genome data
│   ├── dnabert2-output/        # DNABERT2 model outputs
│   ├── download_dir/           # Downloaded files
│   ├── extracted_files/        # Extracted FASTA files
│   ├── gpt2-output/            # GPT-2 model outputs (small/medium/large/ex-large)
│   ├── hf_cache/               # Hugging Face cache
│   ├── logs/                   # Processing logs
│   ├── parquet_files/          # Processed parquet files
│   ├── raw_files/              # Raw text files
│   ├── report/                 # Evaluation reports
│   ├── training_ready_hf_dataset/  # HuggingFace format dataset (train/valid/test)
│   ├── spm_tokenizer.model     # SentencePiece tokenizer model
│   └── spm_tokenizer.vocab     # SentencePiece vocabulary
│
├── logs/                        # Root-level training logs
│
├── molecule_nl/                 # Molecule-related Natural Language data
│   ├── arrow_splits/           # Arrow format split data
│   ├── bert-output/            # BERT model outputs
│   ├── gpt2-output/            # GPT-2 model outputs
│   ├── gpt2_format/            # GPT-2 formatted data
│   ├── logs/                   # Processing logs
│   ├── osunlp/                 # OSU NLP dataset
│   ├── training_ready_hf_dataset/  # HuggingFace format dataset
│   └── molecule_related_natural_language_tokenized.parquet
│
├── protein_sequence/            # Protein sequence data
│   ├── UniRef50/               # UniRef50 reference database
│   ├── bert-output/            # BERT model outputs
│   ├── esm2-output/            # ESM2 model outputs
│   ├── gpt2-output/            # GPT-2 model outputs
│   ├── logs/                   # Processing logs
│   ├── parquet_files/          # Processed parquet files
│   ├── raw_files/              # Raw protein sequences
│   ├── training_ready_hf_dataset/  # HuggingFace format dataset (train/valid/test)
│   └── token_counts.pkl        # Token count statistics
│
└── rna/                         # RNA expression data
    ├── bert-output/            # BERT model outputs
    ├── download_dir/           # Downloaded files
    ├── gpt2-output/            # GPT-2 model outputs
    ├── hf_cache/               # Hugging Face cache
    ├── logs/                   # Processing logs
    ├── loom_dir/               # Loom format files
    ├── metadata_preparation_dir/  # Metadata processing
    ├── parquet_files/          # Processed parquet files
    ├── training_ready_hf_dataset/  # HuggingFace format dataset (train/valid/test)
    ├── gene_list_with_id.tsv   # Gene list with IDs
    ├── gene_list_with_stats.tsv # Gene statistics
    ├── gene_vocab.json         # Gene vocabulary
    ├── rna_stats.json          # RNA statistics
    └── tissue_list.tsv         # Tissue type list
```

## Key Conventions

| Directory Pattern            | Purpose                                                     |
| ---------------------------- | ----------------------------------------------------------- |
| `*-output/`                  | Model training outputs (checkpoints, metrics)               |
| `training_ready_hf_dataset/` | HuggingFace-compatible dataset with train/valid/test splits |
| `logs/`                      | Processing and training logs                                |
| `parquet_files/`             | Tokenized data in Parquet format                            |
| `raw_files/`                 | Pre-processed raw text data                                 |
| `*.marker`                   | Pipeline completion markers                                 |
