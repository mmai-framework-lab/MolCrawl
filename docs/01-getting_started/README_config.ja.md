# パス設定管理

## 概要

このドキュメントでは、プロジェクト全体でデータセットパスを一元管理するための設定ファイルを説明します。

## 設定ファイル

### Python 設定

- `molcrawl/config/paths.py`: Python スクリプト向けのパス定数

### Shell 設定

- `molcrawl/core/env.sh`: Shell スクリプト向けの環境変数設定

## 使い方

### Python スクリプトでの使用

```python
#!/usr/bin/env python3
from molcrawl.core.paths import UNIPROT_DATASET_DIR, REFSEQ_DATASET_DIR

# データセットを読み込む
from datasets import load_from_disk
dataset = load_from_disk(UNIPROT_DATASET_DIR)
```

### Shell スクリプトでの使用

```bash
#!/bin/bash

# 設定を読み込む
source molcrawl/core/env.sh

# LEARNING_SOURCE_DIR からパスを組み立てる
UNIPROT_DATASET_DIR="$LEARNING_SOURCE_DIR/protein_sequence/training_ready_hf_dataset"
REFSEQ_DATASET_DIR="$LEARNING_SOURCE_DIR/genome_sequence/training_ready_hf_dataset"

echo "Learning source: $LEARNING_SOURCE_DIR"
echo "UniProt dataset dir: $UNIPROT_DATASET_DIR"
echo "RefSeq dataset dir: $REFSEQ_DATASET_DIR"
```

## パスの変更方法

データセット保存先ディレクトリを変更するには:

1. `molcrawl/core/env.sh` 内の `LEARNING_SOURCE_DIR` を更新します。
2. 必要に応じて、現在のシェルで `export LEARNING_SOURCE_DIR=...` として上書きします。

例:

```bash
# molcrawl/core/env.sh
export LEARNING_SOURCE_DIR="learning_source"
```

## 利用可能な定数・環境変数

### Python (`paths.py`)

- `LEARNING_SOURCE_DIR`: ベースディレクトリ名
- `PROTEIN_SEQUENCE_DIR`: Protein Sequence ディレクトリパス
- `GENOME_SEQUENCE_DIR`: Genome Sequence ディレクトリパス
- `RNA_DATASET_DIR`: RNA ディレクトリパス
- `MOLECULE_NAT_LANG_DIR`: Molecule_Nat_Lang ディレクトリパス
- `COMPOUNDS_DIR`: Compounds ディレクトリパス
- `UNIPROT_DATASET_DIR`: UniProt データセットパス
- `REFSEQ_DATASET_DIR`: RefSeq データセットパス
- `CELLXGENE_DATASET_DIR`: CellxGene データセットパス
- `COMPOUNDS_DATASET_DIR`: Compounds データセットパス（`organix13_tokenized.parquet` を含む）
- `MOLECULE_NAT_LANG_DATASET_DIR`: Molecule_Nat_Lang データセットパス（`molecule_related_natural_language_tokenized.parquet` を含む）
- `ABSOLUTE_LEARNING_SOURCE_PATH`: ベースディレクトリの絶対パス

### Shell (`env.sh`)

- `$LEARNING_SOURCE_DIR`: ベースディレクトリ名

## 旧 Shell 変数からの対応

`env.sh` が export するのは `$LEARNING_SOURCE_DIR` のみです。
旧変数は以下のように `$LEARNING_SOURCE_DIR` から導出してください。

- `$UNIPROT_DATASET_DIR` -> `$LEARNING_SOURCE_DIR/protein_sequence/training_ready_hf_dataset`
- `$REFSEQ_DATASET_DIR` -> `$LEARNING_SOURCE_DIR/genome_sequence/training_ready_hf_dataset`
- `$CELLXGENE_DATASET_DIR` -> `$LEARNING_SOURCE_DIR/rna/training_ready_hf_dataset`
- `$COMPOUNDS_DATASET_DIR` -> `$LEARNING_SOURCE_DIR/compounds/organix13/compounds/training_ready_hf_dataset`
- `$MOLECULE_NAT_LANG_DATASET_DIR` -> `$LEARNING_SOURCE_DIR/molecule_nat_lang/training_ready_hf_dataset`
