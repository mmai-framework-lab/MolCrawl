# LEARNING_SOURCE_DIR ディレクトリ構造

`LEARNING_SOURCE_DIR` 環境変数は、各生物学的配列タイプごとの学習データ、モデル出力、ログを格納するメインデータディレクトリを指します。

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
├── molecule_nat_lang/                 # Molecule-related Natural Language data
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
│   ├── <dataset_name>/         # UniRef50 / UniRef90 / UniParc などの入力データ
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

## 主要な命名規約

| ディレクトリパターン         | 用途                                                        |
| ---------------------------- | ----------------------------------------------------------- |
| `*-output/`                  | モデル学習出力（チェックポイント、メトリクス）              |
| `training_ready_hf_dataset/` | train/valid/test 分割を持つ HuggingFace 互換データセット    |
| `logs/`                      | 処理・学習ログ                                              |
| `parquet_files/`             | Parquet 形式のトークナイズ済みデータ                        |
| `raw_files/`                 | 前処理済み生テキストデータ                                  |
| `*.marker`                   | パイプライン完了マーカー                                    |

## ディレクトリ生成マッピング

以下のマッピングは `less` での可読性を高めるため、固定幅カラムを使用しています。
カラム幅:

- `Directory`: 47 文字
- `Program`: 64 文字
- `Function`: 116 列目から開始

### compounds

```text
Directory                                        Program                                                           Function
-----------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------
compounds/                                       molcrawl/data/compounds/preparation.py              main()
compounds/data/                                  molcrawl/data/compounds/preparation.py              download_datasets_individually()
compounds/data/zinc20/                           molcrawl/data/compounds/dataset/organix13/zinc/download_and_conve...   download_zinc_files()
compounds/data/opv/                              molcrawl/data/compounds/utils/general.py                               download_opv()
compounds/data/Fraunhofer-SCAI-llamol/           molcrawl/data/compounds/utils/general.py                               download_llamol_datasets()
compounds/benchmark/GuacaMol/                    molcrawl/data/compounds/download_guacamol.py                         download_guacamol()
compounds/compounds_logs/                        molcrawl/core/base.py                                             setup_logging() (called from preparation_script_compounds.py)
compounds/organix13/.../training_ready_hf_data...  molcrawl/data/compounds/dataset/prepare_gpt2_organix13.py            prepare_gpt2_dataset()
compounds/gpt2-output/...                        molcrawl/gpt2/train.py                                            training entry (out_dir)
compounds/bert-output/...                        molcrawl/bert/main.py                                             TrainingArguments(output_dir=model_path)
compounds/chemberta2-output/...                  molcrawl/chemberta2/main.py                                       TrainingArguments(output_dir=model_path)
compounds/logs/                                  workflows/*.sh                                                    mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs
```

### genome_sequence

```text
Directory                                        Program                                                           Function
-----------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------
genome_sequence/download_dir/                    molcrawl/data/genome_sequence/dataset/refseq/download_refseq.py        download_species_refseq()
genome_sequence/extracted_files/                 molcrawl/data/genome_sequence/dataset/refseq/download_refseq.py        extract_file()
genome_sequence/raw_files/                       molcrawl/data/genome_sequence/dataset/refseq/fasta_to_raw.py           fasta_to_raw_genome()
genome_sequence/parquet_files/                   molcrawl/data/genome_sequence/preparation.py        process4_raw_to_parquet()
genome_sequence/hf_cache/                        molcrawl/data/genome_sequence/preparation.py        process5_generate_statistics() (load_dataset cache_dir)
genome_sequence/training_ready_hf_dataset/       molcrawl/data/genome_sequence/dataset/prepare_gpt2.py                  tokenize_batch_dataset()
genome_sequence/spm_tokenizer.*                  molcrawl/data/genome_sequence/dataset/sentence_piece_tokenizer.py      train_tokenizer()
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
protein_sequence/<dataset_name>/                 molcrawl/data/protein_sequence/dataset/uniprot/uniprot_download.py     process_dataset()
protein_sequence/raw_files/                      molcrawl/data/protein_sequence/preparation.py       process2_fasta_to_raw()
protein_sequence/parquet_files/                  molcrawl/data/protein_sequence/dataset/tokenizer.py                    get_parquet_paths() / tokenize_to_parquet()
protein_sequence/training_ready_hf_dataset/      molcrawl/data/protein_sequence/dataset/prepare_gpt2.py                 tokenize_batch_dataset()
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
rna/metadata_preparation_dir/                    molcrawl/data/rna/dataset/cellxgene/script/build_list.py               build_list()
rna/download_dir/                                molcrawl/data/rna/dataset/cellxgene/script/download.py                 download()
rna/loom_dir/                                    molcrawl/data/rna/dataset/cellxgene/script/h5ad_to_loom.py             h5ad_to_loom()
rna/parquet_files/                               molcrawl/data/rna/dataset/tokenization.py                              tokenize()
rna/hf_cache/                                    molcrawl/data/rna/preparation.py                    final statistics step (load_dataset cache_dir)
rna/training_ready_hf_dataset/                   molcrawl/data/rna/dataset/prepare_gpt2.py                              tokenize_batch_dataset()
rna/gpt2-output/...                              molcrawl/gpt2/train.py                                            training entry (out_dir)
rna/bert-output/...                              molcrawl/bert/main.py                                             TrainingArguments(output_dir=model_path)
rna/rnaformer-output/...                         molcrawl/rnaformer/main.py                                        TrainingArguments(output_dir=model_path)
rna/logs/                                        workflows/*.sh                                                    mkdir -p ${LEARNING_SOURCE_DIR}/rna/logs
```

### molecule_nat_lang

```text
Directory                                        Program                                                           Function
-----------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------
molecule_nat_lang/osunlp/SMolInstruct/                 molcrawl/data/molecule_nat_lang/download_smolinstruct.sh                     shell entry (snapshot_download)
molecule_nat_lang/logs/                                molcrawl/preparation/preparation_script_molecule_related_nat_...  main(os.makedirs(logging_dir)) + workflows mkdir -p
molecule_nat_lang/arrow_splits/                        molcrawl/preparation/preparation_script_molecule_related_nat_...  main (Arrow save block)
molecule_nat_lang/gpt2_format/                         molcrawl/preparation/preparation_script_molecule_related_nat_...  main (GPT-2 format save block)
molecule_nat_lang/training_ready_hf_dataset/           molcrawl/molecule_related_nl/dataset/prepare_gpt2.py              tokenize_batch_dataset()
molecule_nat_lang/bert-output/...                      molcrawl/bert/main.py                                             TrainingArguments(output_dir=model_path)
molecule_nat_lang/gpt2-output/...                      molcrawl/gpt2/train.py                                            training entry (out_dir)
```

### 補足

- `LEARNING_SOURCE_DIR/logs`（ルートレベルログ）は、単一の必須パイプラインエントリポイントで必ず作成されるものではなく、主に workflow スクリプトで作成されます。
- 現行コードには genome 抽出出力パスの命名不一致があります（`download_full_refseq.py` で `extracted_files` と `extracted_dir` が混在）。
