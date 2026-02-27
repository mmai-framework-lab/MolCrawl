# Workflow Scripts

Workflow scripts for data preparation, model training, evaluation, and maintenance for the RIKEN Dataset Foundational Model project.

**最終更新**: 2026年1月15日
**スクリプト総数**: 61 (Shell: 60, Python: 1)

## Table of Contents

- [Overview](#-overview)
- [Initial Setup](#-initial-setup)
- [Data Preparation Scripts](#-data-preparation-scripts)
- [Model Training Scripts](#-model-training-scripts)
- [AI Model Evaluation Scripts](#-ai-model-evaluation-scripts)
- [Development & Testing](#-development--testing)
- [Web Interface & Services](#-web-interface--services)
- [Output Structure](#-output-structure)
- [Quick Start Examples](#-quick-start-examples)
- [Prerequisites](#-prerequisites)
- [Script Categories](#-script-categories)
- [統合スクリプトの構造](#-統合スクリプトの構造)
- [Important Notes](#-important-notes)
- [Troubleshooting](#-troubleshooting)
- [Migration Notes](#-migration-notes)

## Overview

This directory contains shell scripts for various project operations including data preparation, model training, evaluation, testing, and system maintenance. All scripts should be executed from the project root directory unless otherwise specified.

The workflow scripts are organized into several categories:

- **Data Preparation** (Phase 01-02): Dataset tokenization and format conversion - 13 scripts
- **Model Training** (Phase 03a-03c): Standard and enhanced training workflows - 28 scripts
- **Model Evaluation**: Comprehensive evaluation with visualization - 8 scripts
- **Development & Testing**: Debugging, batch testing, and validation tools - 4 scripts
- **System Infrastructure**: Web services, experiment tracking, and utilities - 7 scripts
- **Common Library**: Shared utility functions - 1 script

```bash
# Usage pattern
cd /path/to/riken-dataset-fundational-model
./workflows/script_name.sh
```

## 🛠️ Initial Setup

### Environment Setup

| Script        | Purpose                      | Function                                                           |
| ------------- | ---------------------------- | ------------------------------------------------------------------ |
| `00_first.sh` | First-time environment setup | Configure conda channels, create environment, install dependencies |

## 📊 Data Preparation Scripts

このセクションには**13個のデータ準備スクリプト**が含まれています（Phase 1: 6個、Phase 2: 5個、Utility: 2個）

### Phase 1: Dataset Preparation

| Script                             | Purpose                        | Model Type       | Output                          |
| ---------------------------------- | ------------------------------ | ---------------- | ------------------------------- |
| `01_compounds_prepare.sh`          | Compounds dataset tokenization | compounds        | Tokenized SMILES/Scaffolds data |
| `01_compounds-guacamol-prepare.sh` | GuacaMol compounds preparation | compounds        | GuacaMol benchmark data         |
| `01_genome-sequence_prepare.sh`    | Genome sequence data prep      | genome_sequence  | Tokenized genome sequences      |
| `01_molecule-nl_prepare.sh`        | Molecule natural language prep | molecule_nl      | Molecule descriptions           |
| `01_protein-sequence_prepare.sh`   | Protein sequence data prep     | protein_sequence | Tokenized protein sequences     |
| `01_rna_prepare.sh`                | RNA sequence data preparation  | rna              | Tokenized RNA sequences         |

### Phase 2: GPT-2 Data Preparation

| Script                                | Purpose                | Model Type       | Function                |
| ------------------------------------- | ---------------------- | ---------------- | ----------------------- |
| `02-compounds-prepare-gpt2.sh`        | GPT-2 compounds data   | compounds        | Convert to GPT-2 format |
| `02-genome_sequence-prepare-gpt2.sh`  | GPT-2 genome data      | genome_sequence  | Convert to GPT-2 format |
| `02-molecule_nl-prepare-gpt2.sh`      | GPT-2 molecule NL data | molecule_nl      | Convert to GPT-2 format |
| `02-protein_sequence-prepare-gpt2.sh` | GPT-2 protein data     | protein_sequence | Convert to GPT-2 format |
| `02-rna-prepare-gpt2.sh`              | GPT-2 RNA data         | rna              | Convert to GPT-2 format |

### Utility Scripts

| Script                            | Purpose                    | Function                                                |
| --------------------------------- | -------------------------- | ------------------------------------------------------- |
| `common_functions.sh`             | 共通関数ライブラリ         | GPU選択、メモリチェック、環境変数検証などのヘルパー関数 |
| `convert_molecule_nl_to_arrow.sh` | Convert molecule data      | Convert to Arrow format                                 |
| `create_sample_vocab.sh`          | Generate sample vocabulary | Development setup                                       |

## 🏋️ Model Training Scripts

このセクションには**28個のトレーニングスクリプト**が含まれています（Phase 3a: 20個、Phase 3b: 2個、Phase 3c: 5個、Special: 1個）

### Phase 3a: Standard Training

| Script                                   | Purpose            | Model Size | Training Type |
| ---------------------------------------- | ------------------ | ---------- | ------------- |
| `03a-compounds_guacamol-train-small.sh`  | GuacaMol compounds | Small      | Standard      |
| `03a-compounds_guacamol-train-medium.sh` | GuacaMol compounds | Medium     | Standard      |
| `03a-compounds_guacamol-train-large.sh`  | GuacaMol compounds | Large      | Standard      |
| `03a-compounds_guacamol-train-xl.sh`     | GuacaMol compounds | XL         | Standard      |
| `03a-genome_sequence-train-small.sh`     | Genome sequence    | Small      | Standard      |
| `03a-genome_sequence-train-medium.sh`    | Genome sequence    | Medium     | Standard      |
| `03a-genome_sequence-train-large.sh`     | Genome sequence    | Large      | Standard      |
| `03a-genome_sequence-train-xl.sh`        | Genome sequence    | XL         | Standard      |
| `03a-molecule_nl-train-small.sh`         | Molecule NL        | Small      | Standard      |
| `03a-molecule_nl-train-medium.sh`        | Molecule NL        | Medium     | Standard      |
| `03a-molecule_nl-train-large.sh`         | Molecule NL        | Large      | Standard      |
| `03a-molecule_nl-train-xl.sh`            | Molecule NL        | XL         | Standard      |
| `03a-protein_sequence-train-small.sh`    | Protein sequence   | Small      | Standard      |
| `03a-protein_sequence-train-medium.sh`   | Protein sequence   | Medium     | Standard      |
| `03a-protein_sequence-train-large.sh`    | Protein sequence   | Large      | Standard      |
| `03a-protein_sequence-train-xl.sh`       | Protein sequence   | XL         | Standard      |
| `03a-rna-train-small.sh`                 | RNA sequence       | Small      | Standard      |
| `03a-rna-train-medium.sh`                | RNA sequence       | Medium     | Standard      |
| `03a-rna-train-large.sh`                 | RNA sequence       | Large      | Standard      |
| `03a-rna-train-xl.sh`                    | RNA sequence       | XL         | Standard      |

### Phase 3b: Enhanced Training

| Script                                     | Purpose                         | Enhancement                  |
| ------------------------------------------ | ------------------------------- | ---------------------------- |
| `03b-genome_sequence-train-wandb-small.sh` | Genome training with monitoring | Weights & Biases integration |
| `03b-rna-train-yigarashi_refined-small.sh` | Refined RNA training            | Yigarashi method             |

### Phase 3c: BERT Model Training

| Script                                     | Purpose                   | Model Type |
| ------------------------------------------ | ------------------------- | ---------- |
| `03c-compounds-train-bert-small.sh`        | Compounds BERT training   | Small      |
| `03c-genome_sequence-train-bert-small.sh`  | Genome BERT training      | Small      |
| `03c-molecule_nl-train-bert-small.sh`      | Molecule NL BERT training | Small      |
| `03c-protein_sequence-train-bert-small.sh` | Protein BERT training     | Small      |
| `03c-rna-train-bert-small.sh`              | RNA BERT training         | Small      |

### Special Training Scripts

| Script                   | Purpose                  | Function                    |
| ------------------------ | ------------------------ | --------------------------- |
| `train_rna_yigarashi.sh` | Alternative RNA training | Yigarashi-specific approach |

## 🚀 AI Model Evaluation Scripts

### BERT Model Evaluations

| Script                              | Purpose                                  | Dataset    | Dataset Size                | Output Location                                                  |
| ----------------------------------- | ---------------------------------------- | ---------- | --------------------------- | ---------------------------------------------------------------- |
| `run_bert_proteingym_evaluation.sh` | BERT protein fitness prediction (統合版) | ProteinGym | 可変                        | `$LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym_*` |
| `run_bert_clinvar_evaluation.sh`    | BERT variant pathogenicity prediction    | ClinVar    | 2000件（陽性1000+陰性1000） | `$LEARNING_SOURCE_DIR/genome_sequence/report/bert_clinvar_*`     |

**Note**:

- BERT ProteinGymスクリプトは、データ準備・評価・可視化の3フェーズを統合した単一スクリプト
- **ClinVarバランスサンプリング**: 病原性（pathogenic）1000件と良性（benign）1000件をランダム抽出してバランスの取れた評価を実現

### GPT-2 Model Evaluations

#### Genome Sequence (ゲノム配列)

| Script                              | Purpose                    | Dataset | Dataset Size                | Default Device | Output Location                                                    |
| ----------------------------------- | -------------------------- | ------- | --------------------------- | -------------- | ------------------------------------------------------------------ |
| `run_gpt2_clinvar_evaluation.sh`    | 病原性バリアント予測       | ClinVar | 2000件（陽性1000+陰性1000） | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/clinvar_*`            |
| `run_gpt2_cosmic_evaluation.sh`     | がん関連バリアント分析     | COSMIC  | サンプル                    | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/cosmic_*`             |
| `run_gpt2_omim_evaluation_dummy.sh` | 遺伝性疾患予測（テスト用） | OMIM    | サンプル                    | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_evaluation`      |
| `run_gpt2_omim_evaluation_real.sh`  | 遺伝性疾患予測（本番用）   | OMIM    | 実データ                    | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_real_evaluation` |

**Note**:

- `_dummy.sh`: 開発・テスト用サンプルデータで素早く動作確認
- `_real.sh`: 本番評価用。OMIM公式データベースから実データを取得（認証必要）
- **GPU最適化**: すべてのスクリプトはデフォルトでGPU (cuda)を使用（CPUより約4倍高速）
- **既存データ再利用**: `run_gpt2_omim_evaluation_real.sh`は`--existing_omim_dir`オプションでダウンロード済みデータを再利用可能
- **ClinVarバランスサンプリング**: 病原性（pathogenic）1000件と良性（benign）1000件をランダム抽出してバランスの取れた評価を実現

#### Protein Sequence (タンパク質配列)

| Script                               | Purpose                        | Dataset    | Default Model          | Default Device | Output Location                                                            |
| ------------------------------------ | ------------------------------ | ---------- | ---------------------- | -------------- | -------------------------------------------------------------------------- |
| `run_gpt2_proteingym_evaluation.sh`  | タンパク質適応度予測（統合版） | ProteinGym | 指定必須               | GPU (cuda)     | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_proteingym`             |
| `run_gpt2_protein_classification.sh` | タンパク質配列分類（統合版）   | Custom     | protein_sequence-small | GPU (cuda)     | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_protein_classification` |

**Note**:

- **統合スクリプト**: データ準備・評価・可視化の3フェーズを統合した単一スクリプト
- **デフォルトモデル**: `run_gpt2_protein_classification.sh`はモデル指定なしで実行可能（`gpt2-output/protein_sequence-small/ckpt.pt`使用）
- **サンプルデータ作成**: `run_gpt2_proteingym_evaluation.sh --create-sample`で推奨データセットを自動ダウンロード
- **GPU最適化**: デフォルトでGPU使用、`--device cpu`でCPU実行に切り替え可能
- **可視化充実**: 10種類以上のグラフとHTML形式の詳細レポートを自動生成

## 🔧 Development & Testing

### Testing Scripts

| Script                    | Purpose                      | Function                                                                                                            |
| ------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `batch_test_gpt2.sh`      | GPT-2モデル一括テスト        | 複数ドメイン（compounds, molecule_nl, genome, protein_sequence, rna）のチェックポイントを自動検索して一括テスト実行 |
| `gpt2_test_checkpoint.sh` | GPT-2 checkpoint validation  | Model checkpoint testing                                                                                            |
| `debug_protein_bert.sh`   | BERT protein model debugging | Troubleshooting training issues                                                                                     |

### System Utilities

| Script                  | Purpose                | Function                  |
| ----------------------- | ---------------------- | ------------------------- |
| `reboot-cause-check.sh` | System reboot analysis | Infrastructure monitoring |

## 🏗️ Web Interface & Services

このセクションには**5個のスクリプト**が含まれています（Web: 2個、Experiment Management: 3個）

### Web Interface

| Script                | Purpose                 | Function                          | Port/Service  |
| --------------------- | ----------------------- | --------------------------------- | ------------- |
| `web.sh`              | Launch web interface    | Dataset browser and visualization | Default: 3001 |
| `start_api_server.py` | Web API for experiments | RESTful service                   | Default: 8000 |

### Experiment Management

| Script                       | Purpose                        | Function              |
| ---------------------------- | ------------------------------ | --------------------- |
| `setup_experiment_system.sh` | Initialize experiment tracking | System configuration  |
| `start_experiment_system.sh` | Launch experiment services     | Service orchestration |
| `demo_experiment_system.sh`  | System demonstration           | Testing & validation  |

## 📊 Output Structure

All evaluation scripts use the structured directory format and support custom output directories.

### 環境変数

評価スクリプトは以下の環境変数を使用します：

| 環境変数                | 目的                                   | デフォルト             | 必須 |
| ----------------------- | -------------------------------------- | ---------------------- | ---- |
| `LEARNING_SOURCE_DIR`   | 入力データディレクトリ（読み取り専用） | -                      | ✅   |
| `EVALUATION_OUTPUT_DIR` | 出力データディレクトリ（書き込み可能） | `$LEARNING_SOURCE_DIR` | ❌   |

**使用例**：

```bash
# 初回セットアップ
./workflows/00_first.sh

# データ準備の基本フロー
export LEARNING_SOURCE_DIR=/data/learning_source

# Phase 1: Dataset preparation
./workflows/01_compounds_prepare.sh
./workflows/01_genome-sequence_prepare.sh
./workflows/01_protein-sequence_prepare.sh
# ... 他のデータセット

# Phase 2: GPT-2 format conversion (if needed)
./workflows/02-compounds-prepare-gpt2.sh
# ... 対応するGPT-2準備スクリプト

# Phase 3: Training (optional)
./workflows/03a-compounds-guacamol-train-small.sh
# ... 対応する訓練スクリプト

# Evaluation (標準的な使用方法)
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# Web interface
./workflows/web.sh

# 入力と出力を分離する場合
export LEARNING_SOURCE_DIR=/readonly/learning_source  # 入力（読み取り専用）
export EVALUATION_OUTPUT_DIR=/writable/outputs        # 出力（書き込み可能）
./workflows/run_bert_clinvar_evaluation.sh --prepare-data
```

### ディレクトリ構造

```
$LEARNING_SOURCE_DIR/                   # 学習データディレクトリ
├── compounds/                          # 化合物データ
│   ├── image/                          # 可視化画像
│   └── data/                           # トークン化済みデータ
├── genome_sequence/                    # ゲノム配列データ
│   ├── image/                          # 可視化画像
│   ├── data/                           # トークン化済みデータ
│   │   ├── clinvar/                    # ClinVarデータ
│   │   ├── cosmic/                     # COSMICデータ
│   │   └── omim/                       # OMIMデータ
│   └── report/                         # 評価結果
│       ├── bert_clinvar_evaluation/
│       ├── clinvar_evaluation/
│       └── cosmic_evaluation/
├── protein_sequence/                   # タンパク質配列データ
│   ├── image/                          # 可視化画像
│   ├── data/                           # トークン化済みデータ
│   └── report/                         # 評価結果
│       ├── bert_proteingym/
│       └── gpt2_proteingym/
├── rna/                                # RNA配列データ
│   ├── image/                          # 可視化画像
│   └── data/                           # トークン化済みデータ
└── molecule_nl/                        # 分子自然言語データ
    ├── image/                          # 可視化画像
    └── data/                           # トークン化済みデータ

# 学習済みモデル出力
gpt2-output/                            # GPT-2モデル出力
├── compounds-small/
├── genome_sequence-small/
├── protein_sequence-small/
└── rna-small/

# 実行ログ
logs/                                   # スクリプト実行ログ
└── *.log                               # 各スクリプトのログファイル
```

### 出力ディレクトリのカスタマイズ

すべての評価スクリプトは`-o`または`--output_dir`オプションで出力先を指定可能です：

```bash
# BERT ProteinGym評価 - カスタム出力先
./workflows/run_bert_proteingym_evaluation.sh \
  --output_dir /custom/path/bert_proteingym_results

# GPT-2 ClinVar評価 - カスタム出力先
./workflows/run_gpt2_clinvar_evaluation.sh \
  --output_dir /custom/path/clinvar_results

# GPT-2 ProteinGym評価 - カスタム出力先
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv \
  -o /custom/path/proteingym_results

# GPT-2 OMIM実データ評価 - カスタム出力先
./workflows/run_gpt2_omim_evaluation_real.sh \
  --output_dir /custom/path/omim_real_results
```

**注意**:

- 出力先を指定しない場合は、デフォルトで`$LEARNING_SOURCE_DIR/{model_type}/report/{evaluation_type}`に保存されます
- データ準備フェーズ（`--data_dir`）とレポート/可視化フェーズ（`--output_dir`）は別々に指定可能
- 可視化結果は`{output_dir}/visualizations/`サブディレクトリに保存されます

### 各評価ディレクトリの内容

- `*_results.json` - 構造化された評価結果
- `*_report.txt` - 人間が読める形式のサマリー
- `*_detailed_results.csv` - サンプルごとの予測結果
- `visualizations/` - 可視化フェーズで生成されたグラフ・チャート

## 🎯 Quick Start Examples

### Standard Evaluations

#### BERT Model Evaluations

```bash
# BERT ProteinGym evaluation (統合版: データ準備→評価→可視化)
./workflows/run_bert_proteingym_evaluation.sh --max_variants 2000 --batch_size 32

# サンプルデータのみ作成
./workflows/run_bert_proteingym_evaluation.sh --sample_only

# 評価のみ実行（データ準備をスキップ）
./workflows/run_bert_proteingym_evaluation.sh --skip_data_prep

# BERT ClinVar評価（バランスサンプリング: 陽性1000件+陰性1000件）
# 初回実行: データ準備から実行
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# データ準備済みの場合: 評価のみ実行
./workflows/run_bert_clinvar_evaluation.sh

# データ再ダウンロード（強制）
./workflows/run_bert_clinvar_evaluation.sh --force-download
```

#### GPT-2 Genome Sequence Evaluations

```bash
# ClinVar評価（バランスサンプリング: 陽性1000件+陰性1000件）
# 初回実行: HuggingFaceからデータダウンロード＆バランスサンプリング
./workflows/run_gpt2_clinvar_evaluation.sh --download --model-size medium

# データ準備済みの場合: 評価のみ実行
./workflows/run_gpt2_clinvar_evaluation.sh --model-size small

# 評価のみ（データ準備スキップ）
./workflows/run_gpt2_clinvar_evaluation.sh --eval-only --model-size medium

# 可視化のみ実行
./workflows/run_gpt2_clinvar_evaluation.sh --visualize-only

# COSMIC評価
./workflows/run_gpt2_cosmic_evaluation.sh --model_size small --batch_size 32

# OMIM評価（サンプルデータ・開発用）
./workflows/run_gpt2_omim_evaluation_dummy.sh --max_samples 50

# OMIM評価（実データ・本番用、認証必要）
./workflows/run_gpt2_omim_evaluation_real.sh --force_download --model_size medium

# OMIM評価（既存データを再利用）
./workflows/run_gpt2_omim_evaluation_real.sh \
  --existing_omim_dir /path/to/downloaded/omim_data \
  --model_size medium
```

#### GPT-2 Protein Sequence Evaluations

```bash
# ProteinGym評価（統合版）
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m gpt2-output/protein_sequence-small/ckpt.pt \
  -d proteingym_data/sample.csv

# サンプルデータ自動作成と評価（推奨データセットをダウンロード）
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m gpt2-output/protein_sequence-small/ckpt.pt \
  --create-sample

# Protein Classification評価（デフォルトモデル使用）
./workflows/run_gpt2_protein_classification.sh -s

# Protein Classification評価（カスタムモデル指定）
./workflows/run_gpt2_protein_classification.sh \
  -m gpt2-output/protein_sequence-medium/ckpt.pt \
  -s

# 可視化のみ実行（評価済みの場合）
./workflows/run_gpt2_protein_classification.sh \
  -s --skip_data_prep --skip_evaluation
```

### Advanced Options

#### フェーズ別実行（GPT-2スクリプト）

```bash
# データ準備のみ
./workflows/run_gpt2_omim_evaluation_dummy.sh --skip_evaluation --skip_visualization

# 評価のみ（データ準備済みの場合）
./workflows/run_gpt2_omim_evaluation_dummy.sh --skip_data_prep --skip_visualization

# 可視化のみ（評価結果がある場合）
./workflows/run_gpt2_omim_evaluation_dummy.sh --skip_data_prep --skip_evaluation
```

#### デバイスとパフォーマンスの調整

```bash
# CPU使用（GPU非搭載環境向け）
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv --device cpu

# バッチサイズとサンプル数の調整（メモリ節約）
./workflows/run_gpt2_clinvar_evaluation.sh \
  --max_samples 200 --batch_size 8

# ProteinGym高速テスト（最大サンプル数制限）
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv --max_samples 100
```

#### データ管理オプション

```bash
# カスタム出力ディレクトリ指定（すべての評価スクリプト共通）
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv -o /custom/output/path

./workflows/run_bert_clinvar_evaluation.sh \
  --output_dir /custom/clinvar/results

./workflows/run_gpt2_omim_evaluation_real.sh \
  --output_dir /custom/omim/results

# データ準備先とレポート出力先を別々に指定
# (一部のスクリプトで --data_dir と --output_dir を個別指定可能)

# OMIM既存データの再利用（ダウンロードスキップ）
./workflows/run_gpt2_omim_evaluation_real.sh \
  --existing_omim_dir /path/to/omim_data

# ProteinGymサンプルデータの自動作成
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt --create-sample
```

### Experiment System

```bash
# Complete system setup
./workflows/setup_experiment_system.sh

# Start all services
./workflows/start_experiment_system.sh

# Demo the system
./workflows/demo_experiment_system.sh
```

### Development Workflow

```bash
# Batch test all GPT-2 checkpoints
./workflows/batch_test_gpt2.sh gpt2-output/

# Test specific GPT-2 checkpoint
./workflows/gpt2_test_checkpoint.sh

# Debug BERT training
./workflows/debug_protein_bert.sh

# Create development vocabularies
./workflows/create_sample_vocab.sh
```

## 🔧 Prerequisites

### Common Functions Library

`common_functions.sh` provides shared utility functions used across multiple bootstrap scripts:

**主な機能**:

- `check_learning_source_dir()` - LEARNING_SOURCE_DIR環境変数の検証
- `select_best_gpu()` - 最も空きメモリが多いGPUを自動選択
- `check_gpu_memory(gpu_id, min_memory_gb)` - GPU空きメモリの確認
- その他のエラーハンドリングとログ機能

**使用例**:

```bash
# 他のスクリプトから読み込み
source "$(dirname "$0")/common_functions.sh"

# 環境変数チェック
check_learning_source_dir

# 最適なGPUを選択
BEST_GPU=$(select_best_gpu)
export CUDA_VISIBLE_DEVICES=$BEST_GPU
```

### Environment Setup

```bash
# Required environment variables
export LEARNING_SOURCE_DIR=/path/to/learning_source_202508
export CUDA_VISIBLE_DEVICES=0  # For GPU usage

# Load project configuration
source molcrawl/config/env.sh
```

### Dependencies

- Python 3.8+ with transformers, torch, pandas, numpy
- CUDA-capable GPU for model training/evaluation
- Sufficient disk space for datasets and results
- Access to model checkpoints in appropriate directories

## 📝 Script Categories

このディレクトリには61個のスクリプト（Shell: 60, Python: 1）が含まれています：

### 🔍 **Evaluation Scripts** (8 scripts)

自動化されたモデル評価スクリプト（データ準備・評価・可視化の3フェーズ統合）

**BERT Models:**

- `run_bert_proteingym_evaluation.sh` - BERT ProteinGym評価
- `run_bert_clinvar_evaluation.sh` - BERT ClinVar評価

**GPT-2 Genome Sequence:**

- `run_gpt2_clinvar_evaluation.sh` - GPT-2 ClinVar評価
- `run_gpt2_cosmic_evaluation.sh` - GPT-2 COSMIC評価
- `run_gpt2_omim_evaluation_dummy.sh` - GPT-2 OMIM評価（サンプル）
- `run_gpt2_omim_evaluation_real.sh` - GPT-2 OMIM評価（実データ）

**GPT-2 Protein Sequence:**

- `run_gpt2_proteingym_evaluation.sh` - GPT-2 ProteinGym評価
- `run_gpt2_protein_classification.sh` - GPT-2 Protein Classification評価

### 🛠️ **Development Scripts** (4 scripts)

デバッグ、テスト、開発用ユーティリティ

- `batch_test_gpt2.sh` - GPT-2チェックポイント一括テスト（全ドメイン対応）
- `gpt2_test_checkpoint.sh` - GPT-2チェックポイント検証
- `debug_protein_bert.sh` - BERTモデルのデバッグ
- `reboot-cause-check.sh` - システムリブート原因の分析

### 🏭 **Infrastructure Scripts** (4 scripts)

システムセットアップ、サービス管理、実験トラッキング基盤

- `setup_experiment_system.sh` - 実験システムの初期化
- `start_experiment_system.sh` - 実験サービスの起動
- `demo_experiment_system.sh` - システムデモンストレーション
- `start_api_server.py` - Web APIサーバー起動

### ⚙️ **Utility Scripts** (2 scripts)

データ準備とプロジェクトセットアップ用ヘルパースクリプト

- `common_functions.sh` - 共通関数ライブラリ（GPU選択、メモリチェック、環境変数検証）
- `create_sample_vocab.sh` - サンプル語彙ファイルの生成

## 🔄 統合スクリプトの構造

### 3フェーズパイプライン

すべての評価スクリプトは以下の3フェーズで構成されています：

1. **データ準備フェーズ** (`--skip_data_prep`でスキップ可能)
   - データセットのダウンロード/生成
   - 前処理とフォーマット変換
   - `$LEARNING_SOURCE_DIR/{model_type}/data/`に保存
   - **カスタマイズ**: 一部スクリプトで`--data_dir`オプション使用可能

2. **モデル評価フェーズ** (`--skip_evaluation`でスキップ可能)
   - 訓練済みモデルのロード
   - データセットでの推論実行
   - メトリクス計算と結果保存
   - **カスタマイズ**: すべてのスクリプトで`-o`または`--output_dir`使用可能

3. **可視化フェーズ** (`--skip_visualization`でスキップ可能)
   - 評価結果のグラフ生成
   - HTMLレポート作成
   - `{output_dir}/visualizations/`サブディレクトリに保存
   - **カスタマイズ**: 可視化スクリプトで`--output_dir`使用可能

### 出力ディレクトリの柔軟な指定

すべての評価スクリプトで出力先をカスタマイズ可能：

```bash
# デフォルト出力先（LEARNING_SOURCE_DIR配下）
./workflows/run_bert_proteingym_evaluation.sh
# → $LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym_YYYYMMDD_HHMMSS/

# カスタム出力先を指定
./workflows/run_bert_proteingym_evaluation.sh \
  --output_dir /mnt/results/my_proteingym_eval
# → /mnt/results/my_proteingym_eval/

# 相対パス指定も可能
./workflows/run_gpt2_clinvar_evaluation.sh \
  -o ./my_clinvar_results
# → ./my_clinvar_results/

# データ準備とレポート出力を別々に指定（一部スクリプト）
./workflows/run_gpt2_omim_evaluation_real.sh \
  --output_dir /results/omim_eval \
  --config /custom/config.yaml
```

**出力先のデフォルト値:**

| スクリプト | デフォルト出力先 |
|-----------|----------------|
| `run_bert_clinvar_evaluation.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/bert_clinvar_evaluation` |
| `run_bert_proteingym_evaluation.sh` | `$LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym` |
| `run_gpt2_clinvar_evaluation.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/clinvar_evaluation` |
| `run_gpt2_cosmic_evaluation.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/cosmic_evaluation` |
| `run_gpt2_omim_evaluation_dummy.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_evaluation` |
| `run_gpt2_omim_evaluation_real.sh` | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_real_evaluation` |
| `run_gpt2_proteingym_evaluation.sh` | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_proteingym` |
| `run_gpt2_protein_classification.sh` | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_protein_classification` |

### フェーズ別実行の利点

- **開発効率**: データ準備は1回だけ、評価と可視化を繰り返し実行可能
- **デバッグ容易性**: 各フェーズを個別にテスト可能
- **リソース管理**: 必要なフェーズのみ実行してリソースを節約
- **柔軟性**: 外部で準備したデータを使用する場合はデータ準備をスキップ

## 🚨 Important Notes

### 実行環境

- **実行場所**: すべてのスクリプトはプロジェクトルートディレクトリから実行
- **LEARNING_SOURCE_DIR**: 必須環境変数。すべての評価スクリプトで使用
- **GPU要件**: CUDA対応GPUが推奨（CPU実行も可能だが遅い）

### データ管理

- **出力管理**: 結果は自動的にタイムスタンプ付きで整理
- **実データアクセス**: `run_gpt2_omim_evaluation_real.sh`はOMIM認証が必要
- **サンプルデータ**: `_dummy.sh`スクリプトは認証不要で開発・テスト可能

### スクリプト構造

- **統合スクリプト**: データ準備・評価・可視化の3フェーズを1つのスクリプトに統合
- **フェーズスキップ**: `--skip_*`オプションで任意のフェーズをスキップ可能
- **エラーハンドリング**: 堅牢なエラーチェックとリカバリー機能

### リソース管理

- **GPUメモリ**: モデルサイズとバッチサイズに応じて変動
- **ディスク容量**: データセットと結果ファイルのサイズを考慮
- **ログ**: すべての操作で包括的なログを提供
- **パフォーマンス**: GPU使用でCPUより約4倍高速（例: ProteinGym 50サンプル/GPU ≈ 12秒）

### パフォーマンス最適化

- **デフォルトデバイス**: すべての評価スクリプトはGPU (cuda)をデフォルト使用
- **CPU切り替え**: `--device cpu`オプションでCPU実行可能（低速）
- **サンプル数制限**: `--max_samples N`でテスト実行を高速化
- **バッチサイズ調整**: `--batch_size N`でメモリ使用量を制御
- **データ再利用**: `--existing_omim_dir`でダウンロード時間を節約

### 新機能

- **Protein Classification可視化**: 10種類以上のグラフとHTML詳細レポートを自動生成
- **ProteinGymサンプルデータ**: `--create-sample`で推奨データセットを自動ダウンロード
- **OMIM既存データ再利用**: `--existing_omim_dir`でダウンロード済みデータを活用
- **デフォルトモデル**: Protein Classificationはモデル指定なしで実行可能
- **ClinVarバランスサンプリング**: 病原性・良性が1000件ずつのバランスの取れたデータセットで正確な評価

### ClinVarバランスサンプリングの詳細

#### 背景

従来のClinVarデータ準備では数件しか抽出されず、評価の信頼性が低い問題がありました。

#### 改善点

`extract_random_clinvar_samples.py`を使用して以下を実現：

**データ構成**:

- 病原性（Pathogenic）バリアント: 1000件
- 良性（Benign）バリアント: 1000件
- 合計: 2000件のバランスの取れたデータセット

**サンプリング方法**:

1. HuggingFace DatasetsからClinVarデータを取得
2. Clinical Significanceを自動分類（病原性/良性）
3. 各クラスから1000件ずつランダムサンプリング
4. 参照ゲノムから周辺配列を抽出（flanking領域含む）

**利点**:

- クラス不均衡を解消し、正確な精度評価が可能
- 再現可能なランダムサンプリング（seed=42固定）
- 自動化されたデータ準備フロー

**使用方法**:

```bash
# GPT-2 ClinVar評価
./workflows/run_gpt2_clinvar_evaluation.sh --download

# BERT ClinVar評価
./workflows/run_bert_clinvar_evaluation.sh --prepare-data
```

## 📞 Troubleshooting

### よくある問題と解決方法

1. **環境変数エラー**

   ```bash
   # エラー: LEARNING_SOURCE_DIR環境変数が設定されていません
   export LEARNING_SOURCE_DIR=/path/to/learning_source_202508
   ```

2. **モデルファイルが見つからない**

   ```bash
   # モデルディレクトリを確認
   ls -la gpt2-output/
   ls -la runs_train_bert_*/

   # Protein Classificationはデフォルトモデルを使用
   ./workflows/run_gpt2_protein_classification.sh -s
   # → gpt2-output/protein_sequence-small/ckpt.pt を自動使用
   ```

3. **CUDAエラー**

   ```bash
   # GPU確認
   nvidia-smi

   # CPU使用に切り替え（全スクリプトでサポート）
   ./workflows/run_gpt2_*.sh --device cpu

   # 注意: CPUはGPUより約4倍遅い
   ```

4. **データファイルが見つからない**

   ```bash
   # データ準備フェーズを再実行
   ./workflows/run_gpt2_*.sh --force_download

   # または、データ準備のみ実行
   ./workflows/run_gpt2_*.sh --skip_evaluation --skip_visualization

   # ProteinGymサンプルデータの自動作成
   ./workflows/run_gpt2_proteingym_evaluation.sh \
     -m model.pt --create-sample

   # ClinVarバランスサンプリングデータの作成
   # GPT-2の場合
   ./workflows/run_gpt2_clinvar_evaluation.sh --download
   # BERTの場合
   ./workflows/run_bert_clinvar_evaluation.sh --prepare-data
   ```

5. **OMIM実データアクセスエラー**

   ```bash
   # 設定ファイルに認証URLが正しく設定されているか確認
   cat assets/configs/omim_real_data.yaml

   # サンプルデータで動作確認
   ./workflows/run_gpt2_omim_evaluation_dummy.sh

   # 既存データを再利用（再ダウンロードを避ける）
   ./workflows/run_gpt2_omim_evaluation_real.sh \
     --existing_omim_dir /path/to/omim_data
   ```

6. **Pythonパッケージ不足**

   ```bash
   # 必要なパッケージをインストール
   pip install torch transformers pandas numpy scikit-learn matplotlib seaborn sentencepiece scipy
   ```

7. **ProteinGym評価が遅い**

   ```bash
   # GPUを使用（デフォルト、約4倍高速）
   ./workflows/run_gpt2_proteingym_evaluation.sh -m model.pt -d data.csv

   # サンプル数を制限してテスト
   ./workflows/run_gpt2_proteingym_evaluation.sh \
     -m model.pt -d data.csv --max_samples 100

   # 進捗状況: 50サンプル/GPU ≈ 12秒、2770サンプル/GPU ≈ 11分
   ```

8. **可視化エラー**

   ```bash
   # 評価結果があるか確認
   ls -la $LEARNING_SOURCE_DIR/*/report/*/

   # 可視化のみ再実行
   ./workflows/run_gpt2_*.sh --skip_data_prep --skip_evaluation

   # Protein Classificationの詳細レポート
   # → visualizations/ディレクトリに10種類以上のグラフ + HTML

   # カスタム出力先を指定して可視化
   ./workflows/run_gpt2_proteingym_evaluation.sh \
     --skip_data_prep --skip_evaluation \
     -o /custom/visualization/path
   ```

9. **出力ディレクトリが見つからない**

   ```bash
   # デフォルト出力先を確認
   echo $LEARNING_SOURCE_DIR
   ls -la $LEARNING_SOURCE_DIR/*/report/

   # カスタム出力先を使用した場合
   ls -la /path/to/custom/output/

   # 出力先を明示的に指定して再実行
   ./workflows/run_bert_proteingym_evaluation.sh \
     --output_dir /specific/output/path

   # 最新の評価結果ディレクトリを探す
   find $LEARNING_SOURCE_DIR -type d -name "*proteingym*" -o -name "*clinvar*" | sort
   ```

10. **ClinVarデータが数件しか抽出されない**

```bash
# 問題: 従来の方法では少数のサンプルのみ
# 解決策: バランスサンプリングスクリプトを使用

# GPT-2の場合（2000件のバランスデータを自動生成）
./workflows/run_gpt2_clinvar_evaluation.sh --download

# BERTの場合（2000件のバランスデータを自動生成）
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# データセットの統計を確認
python -c "
import pandas as pd
df = pd.read_csv('$LEARNING_SOURCE_DIR/genome_sequence/data/clinvar/clinvar_evaluation_dataset.csv')
print(f'総サンプル数: {len(df)}')
print(df['ClinicalSignificance'].value_counts())
"
# 期待結果: 病原性 1000件、良性 1000件
```

1. **参照ゲノムファイルが見つからない（ClinVarバランスサンプリング）**

    ```bash
    # 参照ゲノムのダウンロード
    wget -P $LEARNING_SOURCE_DIR/genome_sequence/data/ \
      https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.28_GRCh38.p13/GCA_000001405.28_GRCh38.p13_genomic.fna.gz

    # または既にダウンロード済みの場合はパスを確認
    ls -lh $LEARNING_SOURCE_DIR/genome_sequence/data/GCA_000001405.28_GRCh38.p13_genomic.fna*

    # .gzファイルはそのまま使用可能（スクリプトが自動展開）
    ```

2. **複数のGPT-2チェックポイントをまとめてテストしたい**

    ```bash
    # 全ドメインのチェックポイントを一括テスト
    ./workflows/batch_test_gpt2.sh gpt2-output/

    # 特定のディレクトリ配下のみテスト
    ./workflows/batch_test_gpt2.sh path/to/checkpoints/

    # テスト結果は gpt2_test_results_TIMESTAMP/ に保存
    ls -la gpt2_test_results_*/

    # ドメイン別の結果を確認
    # - compounds: 化合物生成の妥当性
    # - genome_sequence: ゲノム配列の整合性
    # - protein_sequence: タンパク質配列の品質
    # - rna: RNA配列の構造妥当性
    # - molecule_nl: 分子記述テキストの品質
    ```

### ログの確認

各スクリプトは詳細なログを出力します：

- コンソール出力: リアルタイムの進行状況
- `logs/`: システムログ（一部のスクリプト）
- `$OUTPUT_DIR/*_report.txt`: 評価結果の詳細レポート

## 🔄 Migration Notes

### スクリプト構造の変更

これらのスクリプトは以下の変更が行われました：

1. **ファイル名の明確化**
   - GPT-2専用スクリプトに`run_gpt2_`プレフィックスを追加
   - BERT専用スクリプトに`run_bert_`プレフィックスを追加
   - OMIM実データスクリプトに`_real`サフィックスを追加

2. **3フェーズ統合**
   - データ準備、評価、可視化スクリプトを統合
   - フェーズ別スキップオプションを追加

3. **LEARNING_SOURCE_DIR構造の統一**
   - すべてのスクリプトで統一されたディレクトリ構造を使用
   - 環境変数チェックを追加

4. **スクリプトパスの統一**
   - すべてのPython実行パスを`scripts/evaluation/{model_type}/`配下に統一

プロジェクトルートディレクトリから実行する限り、すべての機能は同一です。
