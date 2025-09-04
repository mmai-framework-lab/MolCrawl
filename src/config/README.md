# パス設定管理

## 概要

プロジェクト全体で使用するデータセットパスを一元管理するための設定ファイルです。

## 設定ファイル

### Python用設定
- `src/config/paths.py`: Python スクリプト用のパス定数

### Shell用設定  
- `src/config/env.sh`: Shell スクリプト用の環境変数設定

## 使用方法

### Python スクリプトでの使用

```python
#!/usr/bin/env python3
import sys
sys.path.append('src')

from config.paths import UNIPROT_DATASET_DIR, REFSEQ_DATASET_DIR

# データセットを読み込み
from datasets import load_from_disk
dataset = load_from_disk(UNIPROT_DATASET_DIR)
```

### Shell スクリプトでの使用

```bash
#!/bin/bash

# 設定ファイルを読み込み
source src/config/env.sh

# 環境変数を使用
echo "データセットディレクトリ: $UNIPROT_DATASET_DIR"
```

## パス変更方法

データセット保存先ディレクトリを変更する場合：

1. `src/config/paths.py`の`LEARNING_SOURCE_DIR`を変更
2. `src/config/env.sh`の`LEARNING_SOURCE_DIR`を変更

例：
```python
# src/config/paths.py
LEARNING_SOURCE_DIR = 'learning_source_202509'  # 新しいディレクトリ名
```

```bash
# src/config/env.sh
export LEARNING_SOURCE_DIR="learning_source_202509"
```

## 利用可能な定数・環境変数

### Python (paths.py)
- `LEARNING_SOURCE_DIR`: ベースディレクトリ名
- `UNIPROT_DATASET_DIR`: UniProtデータセットパス
- `REFSEQ_DATASET_DIR`: RefSeqデータセットパス  
- `CELLXGENE_DATASET_DIR`: CellxGeneデータセットパス
- `COMPOUNDS_DATASET_DIR`: Compoundsデータセットパス (organix13_tokenized.parquetを含む)
- `MOLECULE_NL_DATASET_DIR`: Molecule_NLデータセットパス (molecule_related_natural_language_tokenized.parquetを含む)
- `ABSOLUTE_LEARNING_SOURCE_PATH`: ベースディレクトリの絶対パス

### Shell (env.sh)
- `$LEARNING_SOURCE_DIR`: ベースディレクトリ名
- `$UNIPROT_DATASET_DIR`: UniProtデータセットパス
- `$REFSEQ_DATASET_DIR`: RefSeqデータセットパス
- `$CELLXGENE_DATASET_DIR`: CellxGeneデータセットパス
- `$COMPOUNDS_DATASET_DIR`: Compoundsデータセットパス
- `$MOLECULE_NL_DATASET_DIR`: Molecule_NLデータセットパス
