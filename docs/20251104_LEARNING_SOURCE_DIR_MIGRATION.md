# LEARNING_SOURCE_DIR 環境変数移行ガイド

## 概要

データセットダウンロード・加工処理を環境変数`LEARNING_SOURCE_DIR`配下に統一し、Gitリポジトリ直下へのファイル生成を防ぐための変更を実施しました。

## 変更日

2025年11月4日

## 変更内容

### 1. 必須化された環境変数

**LEARNING_SOURCE_DIR環境変数が必須になりました。**

未設定の場合、以下のようなエラーメッセージが表示されます：

```text
ERROR: LEARNING_SOURCE_DIR environment variable is not set.
Please set it before running this script:
  export LEARNING_SOURCE_DIR=/path/to/learning_source
  # or
  LEARNING_SOURCE_DIR=learning_20251104 python <script>
```

### 2. 推奨ディレクトリ構造

```text
$LEARNING_SOURCE_DIR/
├── compounds/
│   ├── data/       # データセット保存先
│   ├── logs/       # ログファイル
│   └── report/     # 評価レポート
├── genome_sequence/
│   ├── data/
│   │   ├── clinvar/
│   │   ├── cosmic/
│   │   └── omim/
│   ├── logs/
│   └── report/
├── protein_sequence/
│   ├── data/
│   │   └── proteingym/
│   ├── logs/
│   └── report/
├── rna/
│   ├── data/
│   ├── logs/
│   └── report/
└── molecule_nl/
    ├── data/
    ├── logs/
    └── report/
```

### 3. 更新されたファイル

#### データ準備スクリプト

| ファイル                                                 | 変更内容                  | 新しいデフォルト出力先                                  |
| -------------------------------------------------------- | ------------------------- | ------------------------------------------------------- |
| `scripts/evaluation/bert/proteingym_data_preparation.py` | LEARNING_SOURCE_DIR必須化 | `$LEARNING_SOURCE_DIR/protein_sequence/data/proteingym` |
| `scripts/evaluation/gpt2/omim_data_preparation.py`       | LEARNING_SOURCE_DIR必須化 | `$LEARNING_SOURCE_DIR/genome_sequence/data/omim`        |
| `scripts/evaluation/gpt2/cosmic_data_preparation.py`     | LEARNING_SOURCE_DIR必須化 | `$LEARNING_SOURCE_DIR/genome_sequence/data/cosmic`      |

#### ユーティリティ

| ファイル                         | 変更内容                                                             |
| -------------------------------- | -------------------------------------------------------------------- |
| `src/utils/evaluation_output.py` | `get_learning_source_dir()`でLEARNING_SOURCE_DIR未設定時にエラー終了 |

### 4. 既存の評価スクリプト

以下のスクリプトは既に`utils.evaluation_output`を使用しており、自動的にLEARNING_SOURCE_DIR必須化の恩恵を受けます：

- `scripts/evaluation/bert/clinvar_evaluation.py`
- `scripts/evaluation/gpt2/clinvar_evaluation.py`
- `scripts/evaluation/bert/molecule_nl_evaluation.py`
- `scripts/evaluation/gpt2/molecule_nl_evaluation.py`
- `scripts/evaluation/bert/proteingym_evaluation.py`
- `scripts/evaluation/gpt2/proteingym_evaluation.py`

## 使用方法

### 環境変数の設定

#### 方法1: エクスポート（推奨）

```bash
export LEARNING_SOURCE_DIR=/path/to/learning_source
# または
export LEARNING_SOURCE_DIR=learning_20251104
```

#### 方法2: インラインで指定

```bash
LEARNING_SOURCE_DIR=learning_20251104 python scripts/evaluation/bert/proteingym_data_preparation.py --sample_only
```

### テスト済みの動作例

```bash
# 1. テスト用ディレクトリ構造を作成
mkdir -p learning_20251104/{protein_sequence,genome_sequence,compounds,rna,molecule_nl}/{data,logs,report}

# 2. ProteinGym サンプルデータ生成
LEARNING_SOURCE_DIR=learning_20251104 python scripts/evaluation/bert/proteingym_data_preparation.py --sample_only

# 3. OMIM サンプルデータ生成
LEARNING_SOURCE_DIR=learning_20251104 python scripts/evaluation/gpt2/omim_data_preparation.py --mode sample --num_samples 50

# 4. COSMIC サンプルデータ生成
LEARNING_SOURCE_DIR=learning_20251104 python scripts/evaluation/gpt2/cosmic_data_preparation.py --create_sample_data --max_samples 30
```

### 出力例

```text
learning_20251104/
├── genome_sequence/
│   ├── data/
│   │   ├── cosmic/
│   │   │   └── cosmic_evaluation_dataset.csv
│   │   └── omim/
│   │       ├── data/
│   │       │   ├── omim_evaluation_dataset.csv
│   │       │   └── omim_metadata.json
│   │       └── logs/
│   │           └── omim_preparation_20251104_173921.log
│   └── logs/
│       └── cosmic_preprocessing_20251104_174011.log
└── protein_sequence/
    ├── data/
    │   └── proteingym/
    │       ├── bert_proteingym_sample.csv
    │       ├── bert_proteingym_sample.json
    │       └── bert_proteingym_statistics.txt
    └── logs/
        └── bert_proteingym_prep_20251104_173845.log
```

## 移行チェックリスト

- [x] データ準備スクリプトでLEARNING_SOURCE_DIR必須化
- [x] デフォルト出力先をモデルタイプ別に整理
- [x] ログファイルもLEARNING_SOURCE_DIR配下に移動
- [x] エラーメッセージの改善
- [x] 動作テストの実施

## 注意事項

1. **既存のスクリプトとの互換性**: 既存の評価スクリプトは`utils.evaluation_output`を通じて自動的に新しい動作を継承します。

2. **ログファイルの場所**: 各スクリプトのログファイルは、`$LEARNING_SOURCE_DIR/<model_type>/logs/`配下に保存されます。

3. **レポート出力**: 評価レポートは引き続き`$LEARNING_SOURCE_DIR/<model_type>/report/`配下にタイムスタンプ付きで保存されます。

4. **後方互換性**: `--output_dir`オプションで明示的にパスを指定すれば、任意の場所に出力できます。

## トラブルシューティング

### Q: 既存のスクリプトが動かなくなった

A: `LEARNING_SOURCE_DIR`環境変数を設定してください：

```bash
export LEARNING_SOURCE_DIR=learning_source
```

### Q: 古いデータはどうなる？

A: 既存のデータ（`./bert_proteingym_data`, `./cosmic_data`等）は影響を受けません。新しい実行のみがLEARNING_SOURCE_DIR配下に保存されます。

### Q: テスト用の環境を作りたい

A: 以下のコマンドで専用のテスト環境を作成できます：

```bash
mkdir -p learning_test/{protein_sequence,genome_sequence,compounds,rna,molecule_nl}/{data,logs,report}
export LEARNING_SOURCE_DIR=learning_test
```

## 今後の展開

- [ ] 既存のハードコードされたパスを持つスクリプトの洗い出しと修正
- [ ] CIテストでのLEARNING_SOURCE_DIR設定の自動化
- [ ] ドキュメントの更新（README.md等）
