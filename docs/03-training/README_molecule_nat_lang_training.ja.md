# Molecule NL 学習ガイド

## 概要

このガイドでは、Molecule NL ワークフローの現状と、BERT / GPT-2 両方におけるデータセット互換性を要約します。

## クイックフロー

1. Molecule NL データを準備する。
2. 学習用 Hugging Face データセットを作成する。
3. BERT または GPT-2 を学習する。
4. 必要に応じて互換性チェックを実行する。

## エンドツーエンド準備

### Step 1: Molecule NL データセットを準備

```bash
export LEARNING_SOURCE_DIR="learning_source"
bash workflows/01-molecule_nat_lang-prepare.sh
```

このステップでは `molcrawl/data/molecule_nat_lang/preparation.py` を実行し、以下を作成します。

- `${LEARNING_SOURCE_DIR}/molecule_nat_lang/molecule_related_natural_language_tokenized.parquet`
- `${LEARNING_SOURCE_DIR}/molecule_nat_lang/arrow_splits/`（分割済みデータセット）
- `${LEARNING_SOURCE_DIR}/molecule_nat_lang/gpt2_format/`（トークンストリーム `.pt` ファイル）

### Step 2: 学習用 HF データセットを作成

```bash
export LEARNING_SOURCE_DIR="learning_source"
bash workflows/02-molecule_nat_lang-prepare-gpt2.sh
```

これにより以下が生成されます。

- `${LEARNING_SOURCE_DIR}/molecule_nat_lang/training_ready_hf_dataset`

現行の BERT / GPT-2 の Molecule NL 設定は、どちらもこの `training_ready_hf_dataset` を `dataset_dir` として使用します。

## 学習

### BERT（Molecule NL）

直接実行:

```bash
python molcrawl/models/bert/main.py molcrawl/models/bert/configs/molecule_nat_lang.py
```

ワークフロースクリプト:

```bash
bash workflows/03c-molecule_nat_lang-train-bert-small.sh
```

### GPT-2（Molecule NL）

直接実行:

```bash
python molcrawl/models/gpt2/train.py molcrawl/models/gpt2/configs/molecule_nat_lang/train_gpt2_config.py
```

ワークフロースクリプト:

```bash
bash workflows/03a-molecule_nat_lang-train-small.sh
```

## データ互換性

Molecule NL の前処理パイプラインは、両モデルが同一の元データセットから学習できるよう設計されています。

主なトークン化フィールドは以下です。

```python
{
  "input_ids": List[int],
  "attention_mask": List[int],
  "output_ids": List[int],
  "labels": List[int],
  "input_text": str,
  "real_input_text": str,
  "task_type": str,
  "valid_sample": bool,
  "input_too_long": bool,
}
```

## 互換性チェック

学習前に次を実行できます。

```bash
export LEARNING_SOURCE_DIR="learning_source"
python molcrawl/preparation/test_molecule_nat_lang_compatibility.py
```

想定される要約:

- `BERT compatibility: PASS`
- `GPT-2 compatibility: PASS`

## トラブルシューティング

### データセットが見つからない

- `LEARNING_SOURCE_DIR` が設定されていることを確認。
- 前処理を再実行:

```bash
bash workflows/01-molecule_nat_lang-prepare.sh
bash workflows/02-molecule_nat_lang-prepare-gpt2.sh
```

### 学習中にメモリ不足

- config の `batch_size` を下げる。
- 実効バッチサイズを維持するため `gradient_accumulation_steps` を増やす。
- GPT-2 では必要に応じて `block_size` を下げる。

### GPU 使用率が低い

- `gradient_accumulation_steps` を適度に増やす。
- GPU 選択やログ設定が組み込まれたワークフロースクリプトを利用する。

## 関連ファイル

- BERT config: `molcrawl/models/bert/configs/molecule_nat_lang.py`
- GPT-2 config: `molcrawl/models/gpt2/configs/molecule_nat_lang/train_gpt2_config.py`
- 互換性テスト: `molcrawl/preparation/test_molecule_nat_lang_compatibility.py`
- データセット比較レポート: `docs/07-reports/molecule_nat_lang_dataset_comparison_report.md`
