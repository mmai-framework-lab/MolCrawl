# ワークフロースクリプト

RIKENデータセット基盤モデルプロジェクトのデータ準備、モデル学習、評価、およびメンテナンスのためのワークフロースクリプト集です。

**最終更新**: 2026年3月28日
**スクリプト総数**: 91 (Shell: 89, Python: 2)

## 目次

- [概要](#-概要)
- [初期セットアップ](#-初期セットアップ)
- [データ準備スクリプト](#-データ準備スクリプト)
- [モデル学習スクリプト](#-モデル学習スクリプト)
- [AIモデル評価スクリプト](#-aiモデル評価スクリプト)
- [開発・テスト](#-開発テスト)
- [Webインターフェース＆サービス](#-webインターフェースサービス)
- [出力構造](#-出力構造)
- [クイックスタート例](#-クイックスタート例)
- [前提条件](#-前提条件)
- [スクリプトカテゴリ](#-スクリプトカテゴリ)
- [統合スクリプトの構造](#-統合スクリプトの構造)
- [重要な注意事項](#-重要な注意事項)
- [トラブルシューティング](#-トラブルシューティング)
- [移行ノート](#-移行ノート)

## 概要

このディレクトリには、データ準備、モデル学習、評価、テスト、システムメンテナンスなど、プロジェクトの各種操作のためのシェルスクリプトが含まれています。特に指定がない限り、すべてのスクリプトはプロジェクトルートディレクトリから実行してください。

ワークフロースクリプトはいくつかのカテゴリに分類されます：

- **データ準備** (Phase 01-02)：データセットのトークン化とフォーマット変換 - 17スクリプト
- **モデル学習** (Phase 03a-03g)：GPT-2、BERT、DNABERT-2、ESM-2、RNAformer、ChemBERTa-2 - 46スクリプト
- **モデル評価**：可視化を含む包括的な評価 - 9スクリプト
- **開発・テスト**：デバッグ、一括テスト、検証ツール - 6スクリプト
- **システムインフラ**：Webサービス、実験トラッキング、ユーティリティ - 4スクリプト
- **共通ライブラリ**：共有ユーティリティ関数 - 1スクリプト

```bash
# 使用パターン
cd /path/to/riken-dataset-fundational-model
./workflows/スクリプト名.sh
```

## 🛠️ 初期セットアップ

### 環境セットアップ

| スクリプト    | 目的                       | 機能                                                                   |
| ------------- | -------------------------- | ---------------------------------------------------------------------- |
| `00-first.sh` | 初回環境セットアップ        | condaチャンネルの設定、環境の作成、依存パッケージのインストール         |

## 📊 データ準備スクリプト

このセクションには**17個のデータ準備スクリプト**が含まれています（Phase 1: 11個、Phase 2: 6個）

### フェーズ1：データセット準備

| スクリプト                                         | 目的                                       | モデル種別        | 出力                            |
| -------------------------------------------------- | ------------------------------------------ | ----------------- | ------------------------------- |
| `01-compounds_prepare.sh`                          | 化合物（OrganiX13）データセットのトークン化 | compounds         | トークン化済みSMILES/足場データ  |
| `01-compounds_chembl-prepare.sh`                   | ChEMBL 36データセットの準備                | compounds         | ChEMBLトークン化済みデータ       |
| `01-compounds_guacamol-prepare.sh`                 | GuacaMol化合物の準備                       | compounds         | GuacaMolベンチマークデータ       |
| `01-genome_sequence-prepare.sh`                    | ゲノム配列（RefSeq）データの準備           | genome_sequence   | トークン化済みゲノム配列         |
| `01-genome_sequence_clinvar-prepare.sh`            | ClinVarバリアントデータセットの準備        | genome_sequence   | ClinVarトークン化済みデータ      |
| `01-molecule_nat_lang-prepare.sh`                  | 分子自然言語（SMolInstruct）               | molecule_nat_lang | 分子記述データ                  |
| `01-molecule_nat_lang_mol_instructions-prepare.sh` | Mol-Instructionsデータセットの準備         | molecule_nat_lang | Mol-Instructionsトークン化済みデータ |
| `01-protein_sequence-prepare.sh`                   | タンパク質配列（UniRef50）データの準備     | protein_sequence  | トークン化済みタンパク質配列     |
| `01-protein_sequence_proteingym-prepare.sh`        | ProteinGym v1.3 DMSデータセットの準備      | protein_sequence  | ProteinGymトークン化済みデータ   |
| `01-rna-prepare.sh`                                | RNA配列（CELLxGENE）データの準備           | rna               | トークン化済みRNA配列            |
| `01-rna_celltype-prepare.sh`                       | 細胞種アノテーションデータセットの準備     | rna               | Geneformer細胞種データ           |

### フェーズ2：GPT-2データ準備

| スクリプト                                   | 目的                             | モデル種別        | 機能                       |
| -------------------------------------------- | -------------------------------- | ----------------- | -------------------------- |
| `02-compounds-prepare-gpt2.sh`               | GPT-2化合物（OrganiX13）データ   | compounds         | GPT-2形式に変換            |
| `02-compounds_organix13-prepare-gpt2.sh`     | GPT-2 OrganiX13データ（代替）    | compounds         | GPT-2形式に変換            |
| `02-genome_sequence-prepare-gpt2.sh`         | GPT-2ゲノムデータ                | genome_sequence   | GPT-2形式に変換            |
| `02-molecule_nat_lang-prepare-gpt2.sh`       | GPT-2分子NLデータ                | molecule_nat_lang | GPT-2形式に変換            |
| `02-protein_sequence-prepare-gpt2.sh`        | GPT-2タンパク質データ            | protein_sequence  | GPT-2形式に変換            |
| `02-rna-prepare-gpt2.sh`                     | GPT-2 RNAデータ                  | rna               | GPT-2形式に変換            |

### ユーティリティスクリプト

| スクリプト                              | 目的                       | 機能                                                                  |
| --------------------------------------- | -------------------------- | --------------------------------------------------------------------- |
| `common_functions.sh`                   | 共通関数ライブラリ         | GPU選択、メモリチェック、環境変数検証などのヘルパー関数              |
| `convert_molecule_nat_lang_to_arrow.sh` | 分子データの変換           | Arrow形式に変換                                                       |
| `create_sample_vocab.sh`                | サンプル語彙ファイルの生成 | 開発環境セットアップ                                                  |

## 🏋️ モデル学習スクリプト

このセクションには**46個のトレーニングスクリプト**が含まれています（Phase 3a: 29個、Phase 3b: 1個、Phase 3c: 11個、Phase 3d: 3個、Phase 3e: 3個、Phase 3f: 3個、Phase 3g: 3個）

### フェーズ3a：標準GPT-2学習

| スクリプト                                                  | 目的                        | モデルサイズ | 学習タイプ |
| ------------------------------------------------------- | --------------------------- | ------------ | ---------- |
| `03a-compounds-train-gpt2-small.sh`                     | 化合物（OrganiX13）GPT-2    | Small        | 標準       |
| `03a-compounds-train-gpt2-medium.sh`                    | 化合物（OrganiX13）GPT-2    | Medium       | 標準       |
| `03a-compounds-train-gpt2-large.sh`                     | 化合物（OrganiX13）GPT-2    | Large        | 標準       |
| `03a-compounds-train-gpt2-xl.sh`                        | 化合物（OrganiX13）GPT-2    | XL           | 標準       |
| `03a-compounds_chembl-train-gpt2-small.sh`              | ChEMBL化合物 GPT-2          | Small        | 標準       |
| `03a-compounds_guacamol-train-small.sh`                 | GuacaMol化合物              | Small        | 標準       |
| `03a-compounds_guacamol-train-medium.sh`                | GuacaMol化合物              | Medium       | 標準       |
| `03a-compounds_guacamol-train-large.sh`                 | GuacaMol化合物              | Large        | 標準       |
| `03a-compounds_guacamol-train-xl.sh`                    | GuacaMol化合物              | XL           | 標準       |
| `03a-genome_sequence-train-small.sh`                    | ゲノム配列（RefSeq）        | Small        | 標準       |
| `03a-genome_sequence-train-medium.sh`                   | ゲノム配列（RefSeq）        | Medium       | 標準       |
| `03a-genome_sequence-train-large.sh`                    | ゲノム配列（RefSeq）        | Large        | 標準       |
| `03a-genome_sequence-train-xl.sh`                       | ゲノム配列（RefSeq）        | XL           | 標準       |
| `03a-genome_sequence_clinvar-train-gpt2-small.sh`       | ClinVarゲノム GPT-2         | Small        | 標準       |
| `03a-molecule_nat_lang-train-small.sh`                  | 分子NL（SMolInstruct）      | Small        | 標準       |
| `03a-molecule_nat_lang-train-medium.sh`                 | 分子NL（SMolInstruct）      | Medium       | 標準       |
| `03a-molecule_nat_lang-train-large.sh`                  | 分子NL（SMolInstruct）      | Large        | 標準       |
| `03a-molecule_nat_lang-train-xl.sh`                     | 分子NL（SMolInstruct）      | XL           | 標準       |
| `03a-molecule_nat_lang_mol_instructions-train-small.sh` | Mol-Instructions GPT-2      | Small        | 標準       |
| `03a-protein_sequence-train-small.sh`                   | タンパク質配列（UniRef50）  | Small        | 標準       |
| `03a-protein_sequence-train-medium.sh`                  | タンパク質配列（UniRef50）  | Medium       | 標準       |
| `03a-protein_sequence-train-large.sh`                   | タンパク質配列（UniRef50）  | Large        | 標準       |
| `03a-protein_sequence-train-xl.sh`                      | タンパク質配列（UniRef50）  | XL           | 標準       |
| `03a-protein_sequence_proteingym-train-gpt2-small.sh`   | ProteinGym GPT-2            | Small        | 標準       |
| `03a-rna-train-small.sh`                                | RNA配列（CELLxGENE）        | Small        | 標準       |
| `03a-rna-train-medium.sh`                               | RNA配列（CELLxGENE）        | Medium       | 標準       |
| `03a-rna-train-large.sh`                                | RNA配列（CELLxGENE）        | Large        | 標準       |
| `03a-rna-train-xl.sh`                                   | RNA配列（CELLxGENE）        | XL           | 標準       |
| `03a-rna_celltype-train-gpt2-small.sh`                  | 細胞種アノテーション GPT-2  | Small        | 標準       |

### フェーズ3b：拡張学習

| スクリプト                                 | 目的                             | 拡張機能                     |
| ------------------------------------------ | -------------------------------- | ---------------------------- |
| `03b-genome_sequence-train-wandb-small.sh` | モニタリング付きゲノム配列学習   | Weights & Biases 統合        |

### フェーズ3c：BERTモデル学習

| スクリプト                                                   | 目的                              | モデルサイズ |
| ------------------------------------------------------------ | --------------------------------- | ------------ |
| `03c-compounds-train-bert-small.sh`                          | 化合物（OrganiX13）BERT           | Small        |
| `03c-compounds_chembl-train-bert-small.sh`                   | ChEMBL化合物 BERT                 | Small        |
| `03c-compounds_guacamol-train-bert-small.sh`                 | GuacaMol化合物 BERT               | Small        |
| `03c-genome_sequence-train-bert-small.sh`                    | ゲノム配列（RefSeq）BERT          | Small        |
| `03c-genome_sequence_clinvar-train-bert-small.sh`            | ClinVarゲノム BERT                | Small        |
| `03c-molecule_nat_lang-train-bert-small.sh`                  | 分子NL（SMolInstruct）BERT        | Small        |
| `03c-molecule_nat_lang_mol_instructions-train-bert-small.sh` | Mol-Instructions BERT             | Small        |
| `03c-protein_sequence-train-bert-small.sh`                   | タンパク質配列（UniRef50）BERT    | Small        |
| `03c-protein_sequence_proteingym-train-bert-small.sh`        | ProteinGym BERT                   | Small        |
| `03c-rna-train-bert-small.sh`                                | RNA配列（CELLxGENE）BERT          | Small        |
| `03c-rna_celltype-train-bert-small.sh`                       | 細胞種アノテーション BERT         | Small        |

### フェーズ3d：DNABERT-2学習

| スクリプト                                     | 目的                        | モデルサイズ |
| ---------------------------------------------- | --------------------------- | ------------ |
| `03d-genome_sequence-train-dnabert2-small.sh`  | ゲノム配列 DNABERT-2        | Small        |
| `03d-genome_sequence-train-dnabert2-medium.sh` | ゲノム配列 DNABERT-2        | Medium       |
| `03d-genome_sequence-train-dnabert2-large.sh`  | ゲノム配列 DNABERT-2        | Large        |

### フェーズ3e：ESM-2学習

| スクリプト                                  | 目的                       | モデルサイズ |
| ------------------------------------------- | -------------------------- | ------------ |
| `03e-protein_sequence-train-esm2-small.sh`  | タンパク質配列 ESM-2       | Small        |
| `03e-protein_sequence-train-esm2-medium.sh` | タンパク質配列 ESM-2       | Medium       |
| `03e-protein_sequence-train-esm2-large.sh`  | タンパク質配列 ESM-2       | Large        |

### フェーズ3f：RNAformer学習

| スクリプト                          | 目的                     | モデルサイズ |
| ----------------------------------- | ------------------------ | ------------ |
| `03f-rna-train-rnaformer-small.sh`  | RNA配列 RNAformer        | Small        |
| `03f-rna-train-rnaformer-medium.sh` | RNA配列 RNAformer        | Medium       |
| `03f-rna-train-rnaformer-large.sh`  | RNA配列 RNAformer        | Large        |

### フェーズ3g：ChemBERTa-2学習

| スクリプト                                 | 目的                    | モデルサイズ |
| ------------------------------------------ | ----------------------- | ------------ |
| `03g-compounds-train-chemberta2-small.sh`  | 化合物 ChemBERTa-2      | Small        |
| `03g-compounds-train-chemberta2-medium.sh` | 化合物 ChemBERTa-2      | Medium       |
| `03g-compounds-train-chemberta2-large.sh`  | 化合物 ChemBERTa-2      | Large        |

## 🚀 AIモデル評価スクリプト

### BERTモデル評価

| スクリプト                          | 目的                                     | データセット | データセットサイズ           | 出力先                                                           |
| ----------------------------------- | ---------------------------------------- | ------------ | ---------------------------- | ---------------------------------------------------------------- |
| `run_bert_proteingym_evaluation.sh` | BERTタンパク質適応度予測（統合版）       | ProteinGym   | 可変                         | `$LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym_*` |
| `run_bert_clinvar_evaluation.sh`    | BERTバリアント病原性予測                 | ClinVar      | 2000件（陽性1000+陰性1000）  | `$LEARNING_SOURCE_DIR/genome_sequence/report/bert_clinvar_*`     |

**注意**：

- BERT ProteinGymスクリプトは、データ準備・評価・可視化の3フェーズを統合した単一スクリプト
- **ClinVarバランスサンプリング**：病原性（pathogenic）1000件と良性（benign）1000件をランダム抽出してバランスの取れた評価を実現

### GPT-2モデル評価

#### ゲノム配列

| スクリプト                          | 目的                       | データセット | データセットサイズ           | デフォルトデバイス | 出力先                                                             |
| ----------------------------------- | -------------------------- | ------------ | ---------------------------- | -------------- | ------------------------------------------------------------------ |
| `run_gpt2_clinvar_evaluation.sh`    | 病原性バリアント予測       | ClinVar      | 2000件（陽性1000+陰性1000） | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/clinvar_*`            |
| `run_gpt2_cosmic_evaluation.sh`     | がん関連バリアント分析     | COSMIC       | サンプル                    | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/cosmic_*`             |
| `run_gpt2_omim_evaluation_dummy.sh` | 遺伝性疾患予測（テスト用） | OMIM         | サンプル                    | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_evaluation`      |
| `run_gpt2_omim_evaluation_real.sh`  | 遺伝性疾患予測（本番用）   | OMIM         | 実データ                    | GPU (cuda)     | `$LEARNING_SOURCE_DIR/genome_sequence/report/omim_real_evaluation` |

**注意**：

- `_dummy.sh`：開発・テスト用サンプルデータで素早く動作確認
- `_real.sh`：本番評価用。OMIM公式データベースから実データを取得（認証必要）
- **GPU最適化**：すべてのスクリプトはデフォルトでGPU (cuda)を使用（CPUより約4倍高速）
- **既存データ再利用**：`run_gpt2_omim_evaluation_real.sh`は`--existing_omim_dir`オプションでダウンロード済みデータを再利用可能
- **ClinVarバランスサンプリング**：病原性（pathogenic）1000件と良性（benign）1000件をランダム抽出してバランスの取れた評価を実現

#### タンパク質配列

| スクリプト                           | 目的                           | データセット | デフォルトモデル       | デフォルトデバイス | 出力先                                                                     |
| ------------------------------------ | ------------------------------ | ------------ | ---------------------- | -------------- | -------------------------------------------------------------------------- |
| `run_gpt2_proteingym_evaluation.sh`  | タンパク質適応度予測（統合版） | ProteinGym   | 指定必須               | GPU (cuda)     | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_proteingym`             |
| `run_gpt2_protein_classification.sh` | タンパク質配列分類（統合版）   | Custom       | protein_sequence-small | GPU (cuda)     | `$LEARNING_SOURCE_DIR/protein_sequence/report/gpt2_protein_classification` |

**注意**：

- **統合スクリプト**：データ準備・評価・可視化の3フェーズを統合した単一スクリプト
- **デフォルトモデル**：`run_gpt2_protein_classification.sh`はモデル指定なしで実行可能（`gpt2-output/protein_sequence-small/ckpt.pt`使用）
- **サンプルデータ作成**：`run_gpt2_proteingym_evaluation.sh --create-sample`で推奨データセットを自動ダウンロード
- **GPU最適化**：デフォルトでGPU使用、`--device cpu`でCPU実行に切り替え可能
- **可視化充実**：10種類以上のグラフとHTML形式の詳細レポートを自動生成

#### RNA配列

| スクリプト                        | 目的                     | データセット  | デフォルトデバイス | 出力先                                                    |
| --------------------------------- | ------------------------ | ------------- | -------------- | --------------------------------------------------------- |
| `run_rna_benchmark_evaluation.sh` | RNAベンチマーク評価      | RNA Benchmark | GPU (cuda)     | `$LEARNING_SOURCE_DIR/rna/report/rna_benchmark_*`         |

## 🔧 開発・テスト

### テストスクリプト

| スクリプト                | 目的                         | 機能                                                                                                            |
| ------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `batch_test_gpt2.sh`      | GPT-2モデル一括テスト        | 複数ドメイン（compounds, molecule_nat_lang, genome, protein_sequence, rna）のチェックポイントを自動検索して一括テスト実行 |
| `gpt2_test_checkpoint.sh` | GPT-2チェックポイント検証    | モデルチェックポイントのテスト                                                                                  |
| `debug_protein_bert.sh`   | BERTタンパク質モデルのデバッグ | 学習問題のトラブルシューティング                                                                              |

### システムユーティリティ

| スクリプト              | 目的                   | 機能                   |
| ----------------------- | ---------------------- | ---------------------- |
| `reboot-cause-check.sh` | システムリブート分析   | インフラ監視           |

## 🏗️ Webインターフェース＆サービス

このセクションには**5個のスクリプト**が含まれています（Web: 2個、実験管理: 3個）

### Webインターフェース

| スクリプト            | 目的                     | 機能                              | ポート/サービス |
| --------------------- | ------------------------ | --------------------------------- | ------------- |
| `web.sh`              | Webインターフェースの起動 | データセットブラウザと可視化       | デフォルト: 3001 |
| `start_api_server.py` | 実験Web API              | RESTfulサービス                   | デフォルト: 8000 |

### 実験管理

| スクリプト                   | 目的                         | 機能                   |
| ---------------------------- | ---------------------------- | ---------------------- |
| `setup_experiment_system.sh` | 実験トラッキングの初期化     | システム設定           |
| `start_experiment_system.sh` | 実験サービスの起動           | サービスオーケストレーション |
| `demo_experiment_system.sh`  | システムデモンストレーション | テスト・検証           |

## 📊 出力構造

すべての評価スクリプトは構造化されたディレクトリ形式を使用し、カスタム出力ディレクトリをサポートします。

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

# フェーズ1：データセット準備
./workflows/01_compounds_prepare.sh
./workflows/01_genome-sequence_prepare.sh
./workflows/01_protein-sequence_prepare.sh
# ... 他のデータセット

# フェーズ2：GPT-2形式変換（必要な場合）
./workflows/02-compounds-prepare-gpt2.sh
# ... 対応するGPT-2準備スクリプト

# フェーズ3：学習（オプション）
./workflows/03a-compounds-guacamol-train-small.sh
# ... 対応する訓練スクリプト

# 評価（標準的な使用方法）
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# Webインターフェース
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
└── molecule_nat_lang/                  # 分子自然言語データ
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

**注意**：

- 出力先を指定しない場合は、デフォルトで`$LEARNING_SOURCE_DIR/{model_type}/report/{evaluation_type}`に保存されます
- データ準備フェーズ（`--data_dir`）とレポート/可視化フェーズ（`--output_dir`）は別々に指定可能
- 可視化結果は`{output_dir}/visualizations/`サブディレクトリに保存されます

### 各評価ディレクトリの内容

- `*_results.json` - 構造化された評価結果
- `*_report.txt` - 人間が読める形式のサマリー
- `*_detailed_results.csv` - サンプルごとの予測結果
- `visualizations/` - 可視化フェーズで生成されたグラフ・チャート

## 🎯 クイックスタート例

### 標準評価

#### BERTモデル評価

```bash
# BERT ProteinGym評価（統合版：データ準備→評価→可視化）
./workflows/run_bert_proteingym_evaluation.sh --max_variants 2000 --batch_size 32

# サンプルデータのみ作成
./workflows/run_bert_proteingym_evaluation.sh --sample_only

# 評価のみ実行（データ準備をスキップ）
./workflows/run_bert_proteingym_evaluation.sh --skip_data_prep

# BERT ClinVar評価（バランスサンプリング：陽性1000件+陰性1000件）
# 初回実行：データ準備から実行
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# データ準備済みの場合：評価のみ実行
./workflows/run_bert_clinvar_evaluation.sh

# データ再ダウンロード（強制）
./workflows/run_bert_clinvar_evaluation.sh --force-download
```

#### GPT-2ゲノム配列評価

```bash
# ClinVar評価（バランスサンプリング：陽性1000件+陰性1000件）
# 初回実行：データダウンロード＆バランスサンプリング
./workflows/run_gpt2_clinvar_evaluation.sh --download --model-size medium

# データ準備済みの場合：評価のみ実行
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

#### GPT-2タンパク質配列評価

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

### 高度なオプション

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
# （一部のスクリプトで --data_dir と --output_dir を個別指定可能）

# OMIM既存データの再利用（ダウンロードスキップ）
./workflows/run_gpt2_omim_evaluation_real.sh \
  --existing_omim_dir /path/to/omim_data

# ProteinGymサンプルデータの自動作成
./workflows/run_gpt2_proteingym_evaluation.sh \
  -m model.pt --create-sample
```

### 実験システム

```bash
# システム完全セットアップ
./workflows/setup_experiment_system.sh

# 全サービスの起動
./workflows/start_experiment_system.sh

# システムデモ
./workflows/demo_experiment_system.sh
```

### 開発ワークフロー

```bash
# 全GPT-2チェックポイントの一括テスト
./workflows/batch_test_gpt2.sh gpt2-output/

# 特定GPT-2チェックポイントのテスト
./workflows/gpt2_test_checkpoint.sh

# BERT学習のデバッグ
./workflows/debug_protein_bert.sh

# 開発用語彙ファイルの作成
./workflows/create_sample_vocab.sh
```

## 🔧 前提条件

### 共通関数ライブラリ

`common_functions.sh`は複数のブートストラップスクリプトで使用される共有ユーティリティ関数を提供します：

**主な機能**：

- `check_learning_source_dir()` - LEARNING_SOURCE_DIR環境変数の検証
- `select_best_gpu()` - 最も空きメモリが多いGPUを自動選択
- `check_gpu_memory(gpu_id, min_memory_gb)` - GPU空きメモリの確認
- その他のエラーハンドリングとログ機能

**使用例**：

```bash
# 他のスクリプトから読み込み
source "$(dirname "$0")/common_functions.sh"

# 環境変数チェック
check_learning_source_dir

# 最適なGPUを選択
BEST_GPU=$(select_best_gpu)
export CUDA_VISIBLE_DEVICES=$BEST_GPU
```

### 環境セットアップ

```bash
# 必須環境変数
export LEARNING_SOURCE_DIR=/path/to/learning_source_202508
export CUDA_VISIBLE_DEVICES=0  # GPU使用時

# プロジェクト設定の読み込み
source molcrawl/config/env.sh
```

### 依存パッケージ

- Python 3.8+（transformers、torch、pandas、numpy）
- モデル学習・評価用のCUDA対応GPU
- データセットと結果ファイル用の十分なディスク容量
- 適切なディレクトリ内のモデルチェックポイントへのアクセス

## 📝 スクリプトカテゴリ

このディレクトリには91個のスクリプト（Shell: 89、Python: 2）が含まれています：

### 🔍 **評価スクリプト** (9スクリプト)

自動化されたモデル評価スクリプト（データ準備・評価・可視化の3フェーズ統合）

**BERTモデル：**

- `run_bert_proteingym_evaluation.sh` - BERT ProteinGym評価
- `run_bert_clinvar_evaluation.sh` - BERT ClinVar評価

**GPT-2ゲノム配列：**

- `run_gpt2_clinvar_evaluation.sh` - GPT-2 ClinVar評価
- `run_gpt2_cosmic_evaluation.sh` - GPT-2 COSMIC評価
- `run_gpt2_omim_evaluation_dummy.sh` - GPT-2 OMIM評価（サンプル）
- `run_gpt2_omim_evaluation_real.sh` - GPT-2 OMIM評価（実データ）

**GPT-2タンパク質配列：**

- `run_gpt2_proteingym_evaluation.sh` - GPT-2 ProteinGym評価
- `run_gpt2_protein_classification.sh` - GPT-2 Protein Classification評価

**RNA配列：**

- `run_rna_benchmark_evaluation.sh` - RNA Benchmark評価

### 🛠️ **開発スクリプト** (4スクリプト)

デバッグ、テスト、開発用ユーティリティ

- `batch_test_gpt2.sh` - GPT-2チェックポイント一括テスト（全ドメイン対応）
- `gpt2_test_checkpoint.sh` - GPT-2チェックポイント検証
- `debug_protein_bert.sh` - BERTモデルのデバッグ
- `reboot-cause-check.sh` - システムリブート原因の分析

### 🏭 **インフラスクリプト** (4スクリプト)

システムセットアップ、サービス管理、実験トラッキング基盤

- `setup_experiment_system.sh` - 実験システムの初期化
- `start_experiment_system.sh` - 実験サービスの起動
- `demo_experiment_system.sh` - システムデモンストレーション
- `start_api_server.py` - Web APIサーバー起動

### ⚙️ **ユーティリティスクリプト** (2スクリプト)

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
   - **カスタマイズ**：一部スクリプトで`--data_dir`オプション使用可能

2. **モデル評価フェーズ** (`--skip_evaluation`でスキップ可能)
   - 訓練済みモデルのロード
   - データセットでの推論実行
   - メトリクス計算と結果保存
   - **カスタマイズ**：すべてのスクリプトで`-o`または`--output_dir`使用可能

3. **可視化フェーズ** (`--skip_visualization`でスキップ可能)
   - 評価結果のグラフ生成
   - HTMLレポート作成
   - `{output_dir}/visualizations/`サブディレクトリに保存
   - **カスタマイズ**：可視化スクリプトで`--output_dir`使用可能

### 出力ディレクトリの柔軟な指定

すべての評価スクリプトで出力先をカスタマイズ可能：

```bash
# デフォルト出力先（LEARNING_SOURCE_DIR配下）
./workflows/run_bert_proteingym_evaluation.sh
# -> $LEARNING_SOURCE_DIR/protein_sequence/report/bert_proteingym_YYYYMMDD_HHMMSS/

# カスタム出力先を指定
./workflows/run_bert_proteingym_evaluation.sh \
  --output_dir /mnt/results/my_proteingym_eval
# -> /mnt/results/my_proteingym_eval/

# 相対パス指定も可能
./workflows/run_gpt2_clinvar_evaluation.sh \
  -o ./my_clinvar_results
# -> ./my_clinvar_results/

# データ準備とレポート出力を別々に指定（一部スクリプト）
./workflows/run_gpt2_omim_evaluation_real.sh \
  --output_dir /results/omim_eval \
  --config /custom/config.yaml
```

**出力先のデフォルト値：**

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

- **開発効率**：データ準備は1回だけ、評価と可視化を繰り返し実行可能
- **デバッグ容易性**：各フェーズを個別にテスト可能
- **リソース管理**：必要なフェーズのみ実行してリソースを節約
- **柔軟性**：外部で準備したデータを使用する場合はデータ準備をスキップ

## 🚨 重要な注意事項

### 実行環境

- **実行場所**：すべてのスクリプトはプロジェクトルートディレクトリから実行
- **LEARNING_SOURCE_DIR**：必須環境変数。すべての評価スクリプトで使用
- **GPU要件**：CUDA対応GPUが推奨（CPU実行も可能だが遅い）

### データ管理

- **出力管理**：結果は自動的にタイムスタンプ付きで整理
- **実データアクセス**：`run_gpt2_omim_evaluation_real.sh`はOMIM認証が必要
- **サンプルデータ**：`_dummy.sh`スクリプトは認証不要で開発・テスト可能

### スクリプト構造

- **統合スクリプト**：データ準備・評価・可視化の3フェーズを1つのスクリプトに統合
- **フェーズスキップ**：`--skip_*`オプションで任意のフェーズをスキップ可能
- **エラーハンドリング**：堅牢なエラーチェックとリカバリー機能

### リソース管理

- **GPUメモリ**：モデルサイズとバッチサイズに応じて変動
- **ディスク容量**：データセットと結果ファイルのサイズを考慮
- **ログ**：すべての操作で包括的なログを提供
- **パフォーマンス**：GPU使用でCPUより約4倍高速（例：ProteinGym 50サンプル/GPU ≈ 12秒）

### パフォーマンス最適化

- **デフォルトデバイス**：すべての評価スクリプトはGPU (cuda)をデフォルト使用
- **CPU切り替え**：`--device cpu`オプションでCPU実行可能（低速）
- **サンプル数制限**：`--max_samples N`でテスト実行を高速化
- **バッチサイズ調整**：`--batch_size N`でメモリ使用量を制御
- **データ再利用**：`--existing_omim_dir`でダウンロード時間を節約

### 新機能

- **Protein Classification可視化**：10種類以上のグラフとHTML詳細レポートを自動生成
- **ProteinGymサンプルデータ**：`--create-sample`で推奨データセットを自動ダウンロード
- **OMIM既存データ再利用**：`--existing_omim_dir`でダウンロード済みデータを活用
- **デフォルトモデル**：Protein Classificationはモデル指定なしで実行可能
- **ClinVarバランスサンプリング**：病原性・良性が1000件ずつのバランスの取れたデータセットで正確な評価

### ClinVarバランスサンプリングの詳細

#### 背景

従来のClinVarデータ準備では数件しか抽出されず、評価の信頼性が低い問題がありました。

#### 改善点

`extract_random_clinvar_samples.py`を使用して以下を実現：

**データ構成**：

- 病原性（Pathogenic）バリアント：1000件
- 良性（Benign）バリアント：1000件
- 合計：2000件のバランスの取れたデータセット

**サンプリング方法**：

1. HuggingFace DatasetsからClinVarデータを取得
2. Clinical Significanceを自動分類（病原性/良性）
3. 各クラスから1000件ずつランダムサンプリング
4. 参照ゲノムから周辺配列を抽出（flanking領域含む）

**利点**：

- クラス不均衡を解消し、正確な精度評価が可能
- 再現可能なランダムサンプリング（seed=42固定）
- 自動化されたデータ準備フロー

**使用方法**：

```bash
# GPT-2 ClinVar評価
./workflows/run_gpt2_clinvar_evaluation.sh --download

# BERT ClinVar評価
./workflows/run_bert_clinvar_evaluation.sh --prepare-data
```

## 📞 トラブルシューティング

### よくある問題と解決方法

1. **環境変数エラー**

   ```bash
   # エラー：LEARNING_SOURCE_DIR環境変数が設定されていません
   export LEARNING_SOURCE_DIR=/path/to/learning_source
   ```

2. **モデルファイルが見つからない**

   ```bash
   # モデルディレクトリを確認
   ls -la gpt2-output/
   ls -la runs_train_bert_*/

   # Protein Classificationはデフォルトモデルを使用
   ./workflows/run_gpt2_protein_classification.sh -s
   # -> gpt2-output/protein_sequence-small/ckpt.pt を自動使用
   ```

3. **CUDAエラー**

   ```bash
   # GPU確認
   nvidia-smi

   # CPU使用に切り替え（全スクリプトでサポート）
   ./workflows/run_gpt2_*.sh --device cpu

   # 注意：CPUはGPUより約4倍遅い
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

   # 進捗状況：50サンプル/GPU ≈ 12秒、2770サンプル/GPU ≈ 11分
   ```

8. **可視化エラー**

   ```bash
   # 評価結果があるか確認
   ls -la $LEARNING_SOURCE_DIR/*/report/*/

   # 可視化のみ再実行
   ./workflows/run_gpt2_*.sh --skip_data_prep --skip_evaluation

   # Protein Classificationの詳細レポート
   # -> visualizations/ディレクトリに10種類以上のグラフ + HTML

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
# 問題：従来の方法では少数のサンプルのみ
# 解決策：バランスサンプリングスクリプトを使用

# GPT-2の場合（2000件のバランスデータを自動生成）
./workflows/run_gpt2_clinvar_evaluation.sh --download

# BERTの場合（2000件のバランスデータを自動生成）
./workflows/run_bert_clinvar_evaluation.sh --prepare-data

# データセットの統計を確認
python -c "
import pandas as pd
df = pd.read_csv('$LEARNING_SOURCE_DIR/genome_sequence/data/clinvar/clinvar_evaluation_dataset.csv')
print(f'総サンプル数：{len(df)}')
print(df['ClinicalSignificance'].value_counts())
"
# 期待結果：病原性 1000件、良性 1000件
```

11. **参照ゲノムファイルが見つからない（ClinVarバランスサンプリング）**

    ```bash
    # 参照ゲノムのダウンロード
    wget -P $LEARNING_SOURCE_DIR/genome_sequence/data/ \
      https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.28_GRCh38.p13/GCA_000001405.28_GRCh38.p13_genomic.fna.gz

    # または既にダウンロード済みの場合はパスを確認
    ls -lh $LEARNING_SOURCE_DIR/genome_sequence/data/GCA_000001405.28_GRCh38.p13_genomic.fna*

    # .gzファイルはそのまま使用可能（スクリプトが自動展開）
    ```

12. **複数のGPT-2チェックポイントをまとめてテストしたい**

    ```bash
    # 全ドメインのチェックポイントを一括テスト
    ./workflows/batch_test_gpt2.sh gpt2-output/

    # 特定のディレクトリ配下のみテスト
    ./workflows/batch_test_gpt2.sh path/to/checkpoints/

    # テスト結果は gpt2_test_results_TIMESTAMP/ に保存
    ls -la gpt2_test_results_*/

    # ドメイン別の結果を確認
    # - compounds：化合物生成の妥当性
    # - genome_sequence：ゲノム配列の整合性
    # - protein_sequence：タンパク質配列の品質
    # - rna：RNA配列の構造妥当性
    # - molecule_nat_lang：分子記述テキストの品質
    ```

### ログの確認

各スクリプトは詳細なログを出力します：

- コンソール出力：リアルタイムの進行状況
- `logs/`：システムログ（一部のスクリプト）
- `$OUTPUT_DIR/*_report.txt`：評価結果の詳細レポート

## 🔄 移行ノート

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
