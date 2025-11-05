# Bootstrap Scripts

Project initialization, evaluation, and maintenance scripts for the RIKEN Dataset Foundational Model project.

## 📋 Overview

This directory contains shell scripts for various project operations. All scripts should be executed from the project root directory unless otherwise specified.

```bash
# Usage pattern
cd /path/to/riken-dataset-fundational-model
./bootstraps/script_name.sh
```

## 🚀 AI Model Evaluation Scripts

### BERT Model Evaluations
| Script | Purpose | Dataset | Output Location |
|--------|---------|---------|----------------|
| `run_bert_proteingym_evaluation.sh` | BERT protein fitness prediction (統合版) | ProteinGym | `$LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym_*` |
| `run_bert_clinvar_evaluation.sh` | BERT variant pathogenicity prediction | ClinVar | `$LEARNING_SOURCE_DIR/genome_sequence/report/bert_clinvar_*` |

**Note**: BERT ProteinGymスクリプトは、データ準備・評価・可視化の3フェーズを統合した単一スクリプトです。

### GPT-2 Model Evaluations

#### Genome Sequence (ゲノム配列)
| Script | Purpose | Dataset | Data Type | Output Location |
|--------|---------|---------|-----------|----------------|
| `run_gpt2_clinvar_evaluation.sh` | 病原性バリアント予測 | ClinVar | サンプル | `$LEARNING_SOURCE_DIR/genome_sequence/report/clinvar_*` |
| `run_gpt2_cosmic_evaluation.sh` | がん関連バリアント分析 | COSMIC | サンプル | `$LEARNING_SOURCE_DIR/genome_sequence/report/cosmic_*` |
| `run_gpt2_omim_evaluation_dummy.sh` | 遺伝性疾患予測（テスト用） | OMIM | サンプル | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_evaluation` |
| `run_gpt2_omim_evaluation_real.sh` | 遺伝性疾患予測（本番用） | OMIM | 実データ | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_real_evaluation` |

**Note**: 
- `_dummy.sh`: 開発・テスト用サンプルデータで素早く動作確認
- `_real.sh`: 本番評価用。OMIM公式データベースから実データを取得（認証必要）

#### Protein Sequence (タンパク質配列)
| Script | Purpose | Dataset | Output Location |
|--------|---------|---------|----------------|
| `run_gpt2_proteingym_evaluation.sh` | タンパク質適応度予測（統合版） | ProteinGym | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_proteingym` |
| `run_gpt2_protein_classification.sh` | タンパク質配列分類（統合版） | Custom | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_protein_classification` |

**Note**: GPT-2 ProteinGymスクリプトは、データ準備・評価・可視化の3フェーズを統合した単一スクリプトです。

## 🔧 Development & Debugging

### System Debugging
| Script | Purpose | Use Case |
|--------|---------|----------|
| `debug_protein_bert.sh` | BERT protein model debugging | Troubleshooting training issues |
| `reboot-cause-check.sh` | System reboot analysis | Infrastructure monitoring |
| `test_bert_checkpoint.sh` | BERT checkpoint validation | Model testing |

### Development Utilities
| Script | Purpose | Function |
|--------|---------|----------|
| `create_sample_vocab.sh` | Generate sample vocabulary files | Development setup |

## 🏗️ Experiment Management System

### Infrastructure Scripts
| Script | Purpose | Function | Port/Service |
|--------|---------|----------|-------------|
| `setup_experiment_system.sh` | Initialize experiment tracking | System configuration | - |
| `start_experiment_system.sh` | Launch experiment services | Service orchestration | Multiple services |
| `demo_experiment_system.sh` | System demonstration | Testing & validation | Demo mode |
| `start_api_server.py` | Web API for experiments | RESTful service | Default: 8000 |

## 📊 Output Structure

All evaluation scripts use the structured `LEARNING_SOURCE_DIR` format:

```
$LEARNING_SOURCE_DIR/
├── genome_sequence/
│   ├── data/                           # データ準備フェーズの出力
│   │   ├── clinvar/
│   │   ├── cosmic/
│   │   ├── omim/                       # サンプルデータ
│   │   └── omim_real/                  # 実データ（認証必要）
│   └── report/                         # 評価結果フェーズの出力
│       ├── bert_clinvar_YYYYMMDD_HHMMSS/
│       ├── clinvar_evaluation/         # GPT-2 ClinVar
│       ├── cosmic_evaluation/          # GPT-2 COSMIC
│       ├── omim_evaluation/            # GPT-2 OMIM（サンプル）
│       └── omim_real_evaluation/       # GPT-2 OMIM（実データ）
└── protein_sequence/
    ├── data/                           # データ準備フェーズの出力
    │   ├── bert_proteingym/
    │   ├── gpt2_proteingym/
    │   └── protein_classification/
    └── report/                         # 評価結果フェーズの出力
        ├── bert_proteingym_YYYYMMDD_HHMMSS/
        ├── gpt2_proteingym/
        └── gpt2_protein_classification/
```

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
./bootstraps/run_bert_proteingym_evaluation.sh --max_variants 2000 --batch_size 32

# サンプルデータのみ作成
./bootstraps/run_bert_proteingym_evaluation.sh --sample_only

# 評価のみ実行（データ準備をスキップ）
./bootstraps/run_bert_proteingym_evaluation.sh --skip_data_prep

# BERT ClinVar評価
./bootstraps/run_bert_clinvar_evaluation.sh
```

#### GPT-2 Genome Sequence Evaluations
```bash
# ClinVar評価
./bootstraps/run_gpt2_clinvar_evaluation.sh --model_size medium --max_samples 100

# COSMIC評価
./bootstraps/run_gpt2_cosmic_evaluation.sh --model_size small --batch_size 32

# OMIM評価（サンプルデータ・開発用）
./bootstraps/run_gpt2_omim_evaluation_dummy.sh --max_samples 50

# OMIM評価（実データ・本番用、認証必要）
./bootstraps/run_gpt2_omim_evaluation_real.sh --force_download --model_size medium
```

#### GPT-2 Protein Sequence Evaluations
```bash
# ProteinGym評価（統合版）
./bootstraps/run_gpt2_proteingym_evaluation.sh \
  -m gpt2-output/protein_sequence-small/ckpt.pt \
  -d proteingym_data/sample.csv

# サンプルデータ作成と評価
./bootstraps/run_gpt2_proteingym_evaluation.sh \
  -m gpt2-output/protein_sequence-small/ckpt.pt \
  --create-sample --visualize

# Protein Classification評価
./bootstraps/run_gpt2_protein_classification.sh \
  -m gpt2-output/protein_sequence-small/ckpt.pt \
  -s
```

### Advanced Options

#### フェーズ別実行（GPT-2スクリプト）
```bash
# データ準備のみ
./bootstraps/run_gpt2_omim_evaluation_dummy.sh --skip_evaluation --skip_visualization

# 評価のみ（データ準備済みの場合）
./bootstraps/run_gpt2_omim_evaluation_dummy.sh --skip_data_prep --skip_visualization

# 可視化のみ（評価結果がある場合）
./bootstraps/run_gpt2_omim_evaluation_dummy.sh --skip_data_prep --skip_evaluation
```

#### カスタム設定
```bash
# カスタム出力ディレクトリ指定
./bootstraps/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv -o /custom/output/path

# デバイス指定（CPU使用）
./bootstraps/run_gpt2_proteingym_evaluation.sh \
  -m model.pt -d data.csv --device cpu

# バッチサイズとサンプル数の調整
./bootstraps/run_gpt2_clinvar_evaluation.sh \
  --max_samples 200 --batch_size 8
```

### Experiment System
```bash
# Complete system setup
./bootstraps/setup_experiment_system.sh

# Start all services
./bootstraps/start_experiment_system.sh

# Demo the system
./bootstraps/demo_experiment_system.sh
```

### Development Workflow
```bash
# Debug BERT training
./bootstraps/debug_protein_bert.sh

# Test model checkpoints
./bootstraps/test_bert_checkpoint.sh

# Create development vocabularies
./bootstraps/create_sample_vocab.sh
```

## 🔧 Prerequisites

### Environment Setup
```bash
# Required environment variables
export LEARNING_SOURCE_DIR=/path/to/learning_source_202508
export CUDA_VISIBLE_DEVICES=0  # For GPU usage

# Load project configuration
source src/config/env.sh
```

### Dependencies
- Python 3.8+ with transformers, torch, pandas, numpy
- CUDA-capable GPU for model training/evaluation
- Sufficient disk space for datasets and results
- Access to model checkpoints in appropriate directories

## 📝 Script Categories

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

### 🛠️ **Development Scripts** (2 scripts)  
デバッグ、テスト、開発用ユーティリティ
- `debug_protein_bert.sh` - BERTモデルのデバッグ
- `reboot-cause-check.sh` - システムリブート原因の分析

### 🏭 **Infrastructure Scripts** (4 scripts)
システムセットアップ、サービス管理、実験トラッキング基盤
- `setup_experiment_system.sh` - 実験システムの初期化
- `start_experiment_system.sh` - 実験サービスの起動
- `demo_experiment_system.sh` - システムデモンストレーション
- `start_api_server.py` - Web APIサーバー起動

### ⚙️ **Utility Scripts** (1 script)
データ準備とプロジェクトセットアップ用ヘルパースクリプト
- `create_sample_vocab.sh` - サンプル語彙ファイルの生成

## 🔄 統合スクリプトの構造

### 3フェーズパイプライン
すべての評価スクリプトは以下の3フェーズで構成されています：

1. **データ準備フェーズ** (`--skip_data_prep`でスキップ可能)
   - データセットのダウンロード/生成
   - 前処理とフォーマット変換
   - `$LEARNING_SOURCE_DIR/{model_type}/data/`に保存

2. **モデル評価フェーズ** (`--skip_evaluation`でスキップ可能)
   - 訓練済みモデルのロード
   - データセットでの推論実行
   - メトリクス計算と結果保存

3. **可視化フェーズ** (`--skip_visualization`でスキップ可能)
   - 評価結果のグラフ生成
   - HTMLレポート作成
   - `visualizations/`サブディレクトリに保存

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
   ```

3. **CUDAエラー**
   ```bash
   # GPU確認
   nvidia-smi
   
   # CPU使用に切り替え
   ./bootstraps/run_gpt2_*.sh --device cpu
   ```

4. **データファイルが見つからない**
   ```bash
   # データ準備フェーズを再実行
   ./bootstraps/run_gpt2_*.sh --force_download
   
   # または、データ準備のみ実行
   ./bootstraps/run_gpt2_*.sh --skip_evaluation --skip_visualization
   ```

5. **OMIM実データアクセスエラー**
   ```bash
   # 設定ファイルに認証URLが正しく設定されているか確認
   cat configs/omim_real_data.yaml
   
   # サンプルデータで動作確認
   ./bootstraps/run_gpt2_omim_evaluation_dummy.sh
   ```

6. **Pythonパッケージ不足**
   ```bash
   # 必要なパッケージをインストール
   pip install torch transformers pandas numpy scikit-learn matplotlib seaborn sentencepiece
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