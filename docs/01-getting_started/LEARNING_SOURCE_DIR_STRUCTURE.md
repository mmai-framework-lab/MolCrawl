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
│   │   └── compounds/
│   │       └── training_ready_hf_dataset/
│   └── data/
│       └── zinc20/
│           └── zinc_processed.parquet
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
│   ├── <dataset_name>/         # Input data such as UniRef50 / UniRef90 / UniParc
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

## Directory Generation Mapping

The following mapping uses fixed-width columns for `less` readability.
Column widths:
- `Directory`: 47 chars
- `Program`: 64 chars
- `Function`: starts at column 116

### compounds

```text
Directory                                        Program                                                           Function
-----------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------
compounds/                                       molcrawl/preparation/preparation_script_compounds.py              main()
compounds/data/                                  molcrawl/preparation/preparation_script_compounds.py              download_datasets_individually()
compounds/data/zinc20/                           molcrawl/compounds/dataset/organix13/zinc/download_and_conve...   download_zinc_files()
compounds/data/opv/                              molcrawl/compounds/utils/general.py                               download_opv()
compounds/data/Fraunhofer-SCAI-llamol/           molcrawl/compounds/utils/general.py                               download_llamol_datasets()
compounds/benchmark/GuacaMol/                    molcrawl/preparation/download_guacamol.py                         download_guacamol()
compounds/compounds_logs/                        molcrawl/core/base.py                                             setup_logging() (called from preparation_script_compounds.py)
compounds/organix13/.../training_ready_hf_data...  molcrawl/compounds/dataset/prepare_gpt2_organix13.py            prepare_gpt2_dataset()
compounds/gpt2-output/...                        molcrawl/gpt2/train.py                                            training entry (out_dir)
compounds/bert-output/...                        molcrawl/bert/main.py                                             TrainingArguments(output_dir=model_path)
compounds/chemberta2-output/...                  molcrawl/chemberta2/main.py                                       TrainingArguments(output_dir=model_path)
compounds/logs/                                  workflows/*.sh                                                    mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs
```

### genome_sequence

```text
Directory                                        Program                                                           Function
-----------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------
genome_sequence/download_dir/                    molcrawl/genome_sequence/dataset/refseq/download_refseq.py        download_species_refseq()
genome_sequence/extracted_files/                 molcrawl/genome_sequence/dataset/refseq/download_refseq.py        extract_file()
genome_sequence/raw_files/                       molcrawl/genome_sequence/dataset/refseq/fasta_to_raw.py           fasta_to_raw_genome()
genome_sequence/parquet_files/                   molcrawl/preparation/preparation_script_genome_sequence.py        process4_raw_to_parquet()
genome_sequence/hf_cache/                        molcrawl/preparation/preparation_script_genome_sequence.py        process5_generate_statistics() (load_dataset cache_dir)
genome_sequence/training_ready_hf_dataset/       molcrawl/genome_sequence/dataset/prepare_gpt2.py                  tokenize_batch_dataset()
genome_sequence/spm_tokenizer.*                  molcrawl/genome_sequence/dataset/sentence_piece_tokenizer.py      train_tokenizer()
genome_sequence/data/cosmic/                     molcrawl/evaluation/gpt2/cosmic_data_preparation.py               COSMICDataPreparation.__init__()
genome_sequence/data/omim/                       molcrawl/evaluation/gpt2/omim_data_preparation.py                 prepare_omim_data()
genome_sequence/report/...                       molcrawl/utils/evaluation_output.py                               get_evaluation_output_dir()
genome_sequence/gpt2-output/...                  molcrawl/gpt2/train.py                                            training entry (out_dir)
genome_sequence/bert-output/...                  molcrawl/bert/main.py                                             TrainingArguments(output_dir=model_path)
genome_sequence/dnabert2-output/...              molcrawl/dnabert2/main.py                                         TrainingArguments(output_dir=model_path)
genome_sequence/logs/                            workflows/*.sh                                                    mkdir -p ${LEARNING_SOURCE_DIR}/genome_sequence/logs
```

### protein_sequence

```text
Directory                                        Program                                                           Function
-----------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------
protein_sequence/<dataset_name>/                 molcrawl/protein_sequence/dataset/uniprot/uniprot_download.py     process_dataset()
protein_sequence/raw_files/                      molcrawl/preparation/preparation_script_protein_sequence.py       process2_fasta_to_raw()
protein_sequence/parquet_files/                  molcrawl/protein_sequence/dataset/tokenizer.py                    get_parquet_paths() / tokenize_to_parquet()
protein_sequence/training_ready_hf_dataset/      molcrawl/protein_sequence/dataset/prepare_gpt2.py                 tokenize_batch_dataset()
protein_sequence/report/...                      molcrawl/utils/evaluation_output.py                               get_evaluation_output_dir()
protein_sequence/gpt2-output/...                 molcrawl/gpt2/train.py                                            training entry (out_dir)
protein_sequence/bert-output/...                 molcrawl/bert/main.py                                             TrainingArguments(output_dir=model_path)
protein_sequence/esm2-output/...                 molcrawl/esm2/main.py                                             TrainingArguments(output_dir=model_path)
protein_sequence/logs/                           workflows/*.sh                                                    mkdir -p ${LEARNING_SOURCE_DIR}/protein_sequence/logs
```

### rna

```text
Directory                                        Program                                                           Function
-----------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------
rna/metadata_preparation_dir/                    molcrawl/rna/dataset/cellxgene/script/build_list.py               build_list()
rna/download_dir/                                molcrawl/rna/dataset/cellxgene/script/download.py                 download()
rna/loom_dir/                                    molcrawl/rna/dataset/cellxgene/script/h5ad_to_loom.py             h5ad_to_loom()
rna/parquet_files/                               molcrawl/rna/dataset/tokenization.py                              tokenize()
rna/hf_cache/                                    molcrawl/preparation/preparation_script_rna.py                    final statistics step (load_dataset cache_dir)
rna/training_ready_hf_dataset/                   molcrawl/rna/dataset/prepare_gpt2.py                              tokenize_batch_dataset()
rna/gpt2-output/...                              molcrawl/gpt2/train.py                                            training entry (out_dir)
rna/bert-output/...                              molcrawl/bert/main.py                                             TrainingArguments(output_dir=model_path)
rna/rnaformer-output/...                         molcrawl/rnaformer/main.py                                        TrainingArguments(output_dir=model_path)
rna/logs/                                        workflows/*.sh                                                    mkdir -p ${LEARNING_SOURCE_DIR}/rna/logs
```

### molecule_nl

```text
Directory                                        Program                                                           Function
-----------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------
molecule_nl/osunlp/SMolInstruct/                 molcrawl/preparation/download_smolinstruct.sh                     shell entry (snapshot_download)
molecule_nl/logs/                                molcrawl/preparation/preparation_script_molecule_related_nat_...  main(os.makedirs(logging_dir)) + workflows mkdir -p
molecule_nl/arrow_splits/                        molcrawl/preparation/preparation_script_molecule_related_nat_...  main (Arrow save block)
molecule_nl/gpt2_format/                         molcrawl/preparation/preparation_script_molecule_related_nat_...  main (GPT-2 format save block)
molecule_nl/training_ready_hf_dataset/           molcrawl/molecule_related_nl/dataset/prepare_gpt2.py              tokenize_batch_dataset()
molecule_nl/bert-output/...                      molcrawl/bert/main.py                                             TrainingArguments(output_dir=model_path)
molecule_nl/gpt2-output/...                      molcrawl/gpt2/train.py                                            training entry (out_dir)
```
### Notes

- `LEARNING_SOURCE_DIR/logs` (root-level logs) is not created by a single mandatory pipeline entrypoint; it is mainly created by workflow scripts.
- There is a naming mismatch in current code for genome extraction output in one path (`extracted_files` vs `extracted_dir` in `download_full_refseq.py`).
