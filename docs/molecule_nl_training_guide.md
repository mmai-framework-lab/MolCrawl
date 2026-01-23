# Molecule NL Dataset Training Guide

## データ構造の互換性

新しく準備されたMolecule NLデータセットは、**BERTとGPT-2の両方の学習スクリプトと完全に互換性があります**。

### データ形式

Arrow形式で保存されたデータセットには以下のフィールドが含まれます：

```python
{
    'input_ids': List[int],        # トークン化された入力（BERT・GPT-2共通）
    'attention_mask': List[int],   # アテンションマスク（BERT用）
    'output_ids': List[int],       # 出力トークン（GPT-2で input_ids と結合）
    'labels': List[int],           # ラベル（BERT MLM用）
    'input_text': str,             # 元のSMILES文字列
    'real_input_text': str,        # フォーマット済みプロンプト
    'task_type': str,              # タスクタイプ
    'valid_sample': bool,          # サンプルの有効性
    'input_too_long': bool         # 長文フラグ
}
```

---

## BERT学習

### 1. データの準備

```bash
# データセットの準備（まだの場合）
LEARNING_SOURCE_DIR="learning_20251121" bash workflows/01_molecule-nl_prepare.sh
```

### 2. 学習の実行

```bash
# Smallモデルで学習
python bert/main.py bert/molecule_nl_bert_config.py

# または環境変数で指定
LEARNING_SOURCE_DIR="learning_20251121" python bert/main.py bert/molecule_nl_bert_config.py
```

### 3. 設定のカスタマイズ

`bert/molecule_nl_bert_config.py`を編集：

```python
# モデルサイズ
model_size = "small"  # "small", "medium", "large"

# 学習パラメータ
batch_size = 16
max_steps = 100000
learning_rate = 6e-5
max_length = 512  # 最大シーケンス長
```

### データフォーマット

BERTは以下のフィールドを使用：
- ✅ `input_ids`: トークン化された入力
- ✅ `attention_mask`: パディング部分を示すマスク
- ✅ `labels`: MLM（Masked Language Modeling）用のラベル

**重要**: データは既に正しいフォーマットで準備されているため、**追加の前処理は不要**です。

---

## GPT-2学習

### 1. データの準備

BERTと同じデータセットを使用します：

```bash
# データセットの準備（まだの場合）
LEARNING_SOURCE_DIR="learning_20251121" bash workflows/01_molecule-nl_prepare.sh
```

### 2. 学習の実行

```bash
# デフォルト設定で学習
python gpt2/train.py --config=gpt2/molecule_nl_gpt2_config.py

# または環境変数で指定
LEARNING_SOURCE_DIR="learning_20251121" python gpt2/train.py --config=gpt2/molecule_nl_gpt2_config.py
```

### 3. 設定のカスタマイズ

`gpt2/molecule_nl_gpt2_config.py`を編集：

```python
# モデル設定
block_size = 512  # コンテキスト長
n_layer = 12      # レイヤー数
n_embd = 768      # 埋め込み次元

# 学習パラメータ
batch_size = 16
gradient_accumulation_steps = 8
max_iters = 100000
learning_rate = 3e-4
```

### データフォーマット

GPT-2の`PreparedDataset`クラスは自動的に以下を処理：
- ✅ `input_ids`と`output_ids`を結合して連続したシーケンスを作成
- ✅ 可変長シーケンスを`block_size`に合わせてパディング/トランケーション

**重要**: `train.py`の`get_batch()`関数が自動的にパディング処理を行うため、**追加の処理は不要**です。

---

## データ互換性テスト

学習を開始する前に、データの互換性を確認できます：

```bash
LEARNING_SOURCE_DIR="learning_20251121" python scripts/preparation/test_molecule_nl_compatibility.py
```

期待される出力：

```
======================================================================
Summary
======================================================================
BERT compatibility: ✅ PASS
GPT-2 compatibility: ✅ PASS

✅ All tests passed! Data is compatible with both BERT and GPT-2.
```

---

## ディレクトリ構造

```
learning_20251121/
└── molecule_nl/
    ├── arrow_splits/              # 学習用データ（BERT・GPT-2共通）
    │   ├── train.arrow/
    │   ├── test.arrow/
    │   └── valid.arrow/
    └── molecule_related_natural_language_tokenized.parquet  # 統合parquetファイル
```

---

## トラブルシューティング

### データが見つからない

```bash
# データセットを再準備
LEARNING_SOURCE_DIR="learning_20251121" bash workflows/01_molecule-nl_prepare.sh
```

### メモリ不足エラー

**BERT:**
```python
# batch_sizeを減らす
batch_size = 8  # デフォルトは16
gradient_accumulation_steps = 8  # 実効バッチサイズを維持
```

**GPT-2:**
```python
# batch_sizeまたはblock_sizeを減らす
batch_size = 8
block_size = 256  # デフォルトは512
```

### GPU使用率が低い

```python
# gradient_accumulation_stepsを調整
gradient_accumulation_steps = 16  # より大きな実効バッチサイズ
```

---

## データ品質の特徴

新しいデータセットは以下の品質改善が施されています：

✅ **SMILES検証**: 全サンプルでSMILES構造の化学的妥当性を検証  
✅ **無効サンプルの除外**: `valid_sample=True`のサンプルのみを使用  
✅ **長文フラグ**: `input_too_long`で長すぎるサンプルを識別  
✅ **タスクタイプの明示**: 14種類のタスクタイプを`task_type`で管理

---

## 統計情報

- **Train**: 3,267,176サンプル
- **Test**: 30,344サンプル
- **Valid**: 17,781サンプル
- **総トークン数**: 約342M トークン

### タスク分布

| タスク | サンプル数 |
|--------|------------|
| forward_synthesis | 977,920 |
| retrosynthesis | 947,983 |
| name_conversion-* | ~1.2M (4種類) |
| molecule_captioning | 60,305 |
| molecule_generation | 60,260 |
| property_prediction-* | ~51K (7種類) |

---

## 既存コードへの影響

### ✅ 変更不要

- **学習スクリプト**: `bert/main.py`と`gpt2/train.py`は**そのまま使用可能**
- **データローダー**: `PreparedDataset`クラスが自動的に新形式を処理
- **トークナイザー**: Llama-2トークナイザーを継続使用

### ⚠️ 注意が必要

旧データセット（`learning_source_202508`）からの移行時：

- カラム名の変更: `task` → `task_type`
- 削除されたカラム: `sample_id`, `raw_input`, `raw_output`など
- サンプル数の減少: 約0.81%（品質検証により無効サンプルを除外）

詳細は`docs/molecule_nl_dataset_comparison_report.md`を参照してください。

---

## 次のステップ

1. **データの準備**: `bash workflows/01_molecule-nl_prepare.sh`
2. **互換性テスト**: `python scripts/preparation/test_molecule_nl_compatibility.py`
3. **BERT学習開始**: `python bert/main.py bert/molecule_nl_bert_config.py`
4. **GPT-2学習開始**: `python gpt2/train.py --config=gpt2/molecule_nl_gpt2_config.py`

---

## 参考リンク

- [データセット比較レポート](../docs/molecule_nl_dataset_comparison_report.md)
- [SMolInstructデータセット](https://huggingface.co/datasets/osunlp/SMolInstruct)
- [BERT設定例](bert/molecule_nl_bert_config.py)
- [GPT-2設定例](gpt2/molecule_nl_gpt2_config.py)
