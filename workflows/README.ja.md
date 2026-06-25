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
- **モデル学習** (Phase 03a-03g)：GPT-2、BERT、DNABERT-2、ESM-2、ChemBERTa-2 - 43スクリプト
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

### フェーズ3g：ChemBERTa-2学習

| スクリプト                                 | 目的                    | モデルサイズ |
| ------------------------------------------ | ----------------------- | ------------ |
| `03g-compounds-train-chemberta2-small.sh`  | 化合物 ChemBERTa-2      | Small        |
| `03g-compounds-train-chemberta2-medium.sh` | 化合物 ChemBERTa-2      | Medium       |
| `03g-compounds-train-chemberta2-large.sh`  | 化合物 ChemBERTa-2      | Large        |

## 🚀 AIモデル評価スクリプト

評価ハーネスはアーキ非依存の単一レイアウト
（`molcrawl/tasks/evaluation/<task>/`）に移行しました。各タスクは
`__main__.py` CLI と薄い `workflows/eval-<task>.sh` ドライバを持ち、
データ取得は `workflows/data/eval-data-<task>.sh` 配下にあります。
認証必須のダウンロード（COSMIC / OMIM / ゲートされた HuggingFace 等）
はリポジトリ直下の `.env` から読み込まれます。テンプレートは
[`.env.example`](../.env.example) を参照。

| ワークフロー                              | タスクパッケージ                                        | モダリティ        | 備考                                       |
| ---------------------------------------- | ------------------------------------------------------- | ----------------- | ------------------------------------------- |
| `eval-clinvar.sh`                        | `molcrawl.tasks.evaluation.clinvar`                     | genome_sequence   | 陽性/陰性バランスサンプリング + bootstrap CI |
| `eval-gnomad.sh`                         | `molcrawl.tasks.evaluation.gnomad_af_correlation`       | genome_sequence   | AF-bin 層化サンプリング                      |
| `eval-gue.sh`                            | `molcrawl.tasks.evaluation.gue`                         | genome_sequence   | 28 sub-task                                  |
| `eval-proteingym.sh`                     | `molcrawl.tasks.evaluation.proteingym`                  | protein_sequence  | 参照 vs 変異の尤度差                         |
| `eval-deeploc.sh`                        | `molcrawl.tasks.evaluation.deeploc`                     | protein_sequence  | 細胞内局在 10 クラス分類                     |
| `eval-tape.sh`                           | `molcrawl.tasks.evaluation.tape`                        | protein_sequence  | fluorescence / stability / remote_homology / SS3 / SS8 |
| `eval-protein-foldability.sh`            | `molcrawl.tasks.evaluation.protein_foldability`         | protein_sequence  | 構造非依存の foldability 代理指標             |
| `eval-moleculenet.sh`                    | `molcrawl.tasks.evaluation.moleculenet`                 | compounds         | 12 標準サブセット（property prediction）     |
| `eval-moses.sh`                          | `molcrawl.tasks.evaluation.moses`                       | compounds         | validity / uniqueness / novelty / int. div. |
| `eval-chembl-heldout.sh`                 | `molcrawl.tasks.evaluation.chembl_scaffold_heldout`     | compounds         | scaffold-disjoint perplexity                |
| `eval-rna-benchmark.sh`                  | `molcrawl.tasks.evaluation.rna_benchmark`               | rna               | 組織ごとの PLL                               |
| `eval-tabula-sapiens.sh`                 | `molcrawl.tasks.evaluation.tabula_sapiens`              | rna               | cell-type annotation                        |
| `eval-replogle-perturb-seq.sh`           | `molcrawl.tasks.evaluation.replogle_perturb_seq`        | rna               | perturbation response                       |
| `eval-molecule-nat-lang.sh`              | `molcrawl.tasks.evaluation.molecule_nat_lang`           | molecule_nat_lang | molecule / caption ペア尤度                  |
| `eval-chemllmbench.sh`                   | `molcrawl.tasks.evaluation.chemllmbench`                | molecule_nat_lang | 9 sub-task（うち 3 配線済み）                |
| `eval-chebi20.sh`                        | `molcrawl.tasks.evaluation.chebi20`                     | molecule_nat_lang | 双方向生成評価                               |
| (認証必須) `eval-data-cosmic.sh`          | `molcrawl.tasks.evaluation.cosmic`                     | genome_sequence   | COSMIC_EMAIL / COSMIC_PASSWORD              |
| (認証必須) `eval-data-omim.sh`            | `molcrawl.tasks.evaluation.omim`                       | genome_sequence   | OMIM_API_KEY                                |

**共通機能**：

- **Bootstrap 95 % CI** をリサンプリング可能なメトリクスに対し点推定とともに表示
- **予測ログ**：すべての評価器が `predictions.jsonl`（per-row）と
  `predictions.txt`（best/worst-fit narrative）を `metrics.json` /
  `REPORT.md` の隣に出力
- **冪等な matrix runner**：`workflows/eval-matrix-bench.sh` は
  (評価器 × アーキ × サイズ) sweep を駆動し、REPORT.md が既存の combo は skip
- **ダッシュボード**：`python -m molcrawl.tasks.evaluation._dashboard`
  で REPORT.md を巡回し `docs-src/assets/data/evaluations.json` を再生成

リファクタ前のアーキ別ラッパー（`run_bert_*`、`run_gpt2_*`）はこのレイアウト
に置き換わって廃止されました。`protein_classification` タスクも削除済み —
その用途は `proteingym`（変異効果）、`deeploc`（局在）、
`tape.remote_homology`（フォールド分類）でカバーされます。

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

# 評価（アーキ非依存ハーネス）
bash workflows/data/eval-data-clinvar.sh   # データ取得
bash workflows/eval-clinvar.sh             # 評価実行

# Webインターフェース
./workflows/web.sh

# 入力と出力を分離する場合
export LEARNING_SOURCE_DIR=/readonly/learning_source  # 入力（読み取り専用）
export OUTPUT_DIR=/writable/outputs/clinvar           # 評価出力（書き込み可能）
bash workflows/eval-clinvar.sh
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

各評価ラッパーは `OUTPUT_DIR` を自動合成します:

```
${LEARNING_SOURCE_DIR}/experiment_data/eval/<modality>-<arch>-<size>/<RUNTAG>/
```

`<modality>-<arch>-<size>` は `MODEL_PATH` から自動派生。 操作者は
`RUNTAG` だけ指定すれば履歴を分離できます:

```bash
# ClinVar — 出力先は
#   ${LEARNING_SOURCE_DIR}/experiment_data/eval/genome_sequence-bert-small/clinvar_nper1000/
LEARNING_SOURCE_DIR=$LSD \
MODEL_PATH=$LSD/genome_sequence/bert-output/genome_sequence-small/checkpoint-60000 \
CLINVAR_DATA=$LSD/eval/clinvar/clinvar.csv \
RUNTAG=clinvar_nper1000 N_PER_CLASS=1000 \
bash workflows/eval-clinvar.sh

# ProteinGym / MOSES / MoleculeNet / TAPE / GUE / DeepLoc 等も同パターン。
# OUTPUT_DIR を明示的に渡せば slug レイアウトを抜け出せる(必要に応じて)。
```

**注意**:

- model slug は `common_functions.sh` の `derive_model_slug` が
  `MODEL_PATH` から派生(絶対 / 相対パス両方対応)。
- `RUNTAG` 省略時は `<task>_default`(意図的に "未指定の印")。
- smoke run は `_smoke/<model-slug>/<RUNTAG>/` に、
  失敗 run(`metrics.json` 不在)は `_failed/<old_dir>/` に分離。
- 詳細は [`docs/04-evaluation/eval_dashboard.ja.md`](../docs/04-evaluation/eval_dashboard.ja.md) を参照。
- 各評価器が `metrics.json` / `REPORT.md` / `predictions.jsonl` /
  `predictions.txt` を `OUTPUT_DIR` に出力。
- Bootstrap CI は適用可能な metric に対し点推定とともに表示。

### 各評価ディレクトリの内容

- `*_results.json` - 構造化された評価結果
- `*_report.txt` - 人間が読める形式のサマリー
- `*_detailed_results.csv` - サンプルごとの予測結果
- `visualizations/` - 可視化フェーズで生成されたグラフ・チャート

