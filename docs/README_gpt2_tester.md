# GPT2チェックポイント テスト・検証システム

このディレクトリには、GPT2学習スクリプトで生成されたチェックポイントを包括的にテスト・検証するためのツール群が含まれています。

## ファイル構成

- `test_checkpoint.py` - メインのテスト・検証スクリプト
- `test_helper.py` - チェックポイント検索とテスト実行を支援するヘルパー
- `batch_test_gpt2.sh` - 複数チェックポイントの一括テスト用バッチスクリプト
- `test_configs/` - 各ドメイン用のテスト設定ファイル

## 主な機能

### 1. チェックポイントテスト機能

- ✅ カスタムGPT2チェックポイントの読み込み
- ✅ Hugging Face形式への変換
- ✅ パープレキシティ計算
- ✅ Top-1/Top-5精度測定
- ✅ テキスト生成品質評価
- ✅ モデルパフォーマンス統計
- ✅ 包括的なテストレポート生成

### 2. マルチドメイン対応

- 🧪 **compounds** - SMILES化学構造データ
- 🔬 **molecule_nat_lang** - 分子関連自然言語
- 🧬 **genome** - ゲノム配列データ
- 🧪 **protein_sequence** - タンパク質配列
- 🧬 **rna** - RNA配列データ

### 3. トークナイザー統合

各ドメインの特化トークナイザーを自動検出して使用:

- CompoundsTokenizer (SMILES)
- MoleculeNatLangTokenizer (CodeLlama-7b)
- SentencePiece (ゲノム)
- EsmSequenceTokenizer (タンパク質)
- TranscriptomeTokenizer (RNA)

## 使用方法

### 基本的な使用方法

#### 1. 単一チェックポイントのテスト

```bash
# 基本テスト
python gpt2/test_checkpoint.py --checkpoint_path=out-compounds/ckpt.pt

# ドメイン指定テスト
python gpt2/test_checkpoint.py \
    --checkpoint_path=out-compounds/ckpt.pt \
    --domain=compounds \
    --vocab_path=assets/molecules/vocab.txt \
    --convert_to_hf

# カスタムデータセットでのテスト
python gpt2/test_checkpoint.py \
    --checkpoint_path=out-molecule-nl/ckpt.pt \
    --domain=molecule_nat_lang \
    --test_dataset_params='{"dataset_dir": "learning_source"}' \
    --max_test_samples=1000
```

#### 2. ヘルパースクリプトを使った便利なテスト

```bash
# 利用可能なチェックポイントをリストアップ
python gpt2/test_helper.py --list_only

# 特定ディレクトリ内のチェックポイントを検索
python gpt2/test_helper.py --search_dir=out-compounds --list_only

# 自動実行
python gpt2/test_helper.py --search_dir=out-compounds --auto_run
```

#### 3. 一括テスト（推奨）

```bash
# 現在のディレクトリ内の全チェックポイントを一括テスト
./gpt2/batch_test_gpt2.sh

# 特定ディレクトリ内のチェックポイントを一括テスト
./gpt2/batch_test_gpt2.sh /path/to/checkpoints
```

### 高度な使用例

#### ドメイン特化テスト

```bash
# Compoundsドメインの完全テスト
python gpt2/test_checkpoint.py \
    --checkpoint_path=out-compounds/ckpt.pt \
    --domain=compounds \
    --vocab_path=assets/molecules/vocab.txt \
    --test_dataset_params='{"dataset_dir": "learning_source"}' \
    --convert_to_hf \
    --max_test_samples=2000 \
    --output_dir=comprehensive_compounds_test

# Protein sequenceドメインテスト
python gpt2/test_checkpoint.py \
    --checkpoint_path=out-protein/ckpt.pt \
    --domain=protein_sequence \
    --test_dataset_params='{"dataset_dir": "outputs/protein_sequence/training_ready_hf_dataset"}' \
    --convert_to_hf
```

#### Hugging Face形式への変換のみ

```bash
python gpt2/test_checkpoint.py \
    --checkpoint_path=out-compounds/ckpt.pt \
    --convert_to_hf \
    --output_dir=hf_converted_model
```

## 出力結果

### テストレポート

各テスト実行後、以下の形式のJSONレポートが生成されます:

```json
{
  "checkpoint_path": "out-compounds/ckpt.pt",
  "test_timestamp": "2025-08-01 10:30:45",
  "results": {
    "checkpoint_info": {
      "iter_num": 6000,
      "best_val_loss": 2.85,
      "config": {...}
    },
    "performance_stats": {
      "total_params": 124000000,
      "vocab_size": 32024,
      "block_size": 1024,
      "n_layer": 12,
      "n_head": 12,
      "n_embd": 768
    },
    "perplexity": 15.23,
    "avg_loss": 2.72,
    "accuracy": 0.3245,
    "top5_accuracy": 0.6789,
    "generated_samples": [...],
    "status": "success"
  }
}
```

### 一括テスト結果

```text
gpt2_test_results_20250801_103045/
├── compounds_out-compounds/
│   ├── gpt2_test_report.json
│   └── hf_model/
├── molecule_nat_lang_out-molecule-nl/
│   ├── gpt2_test_report.json
│   └── hf_model/
└── test_summary.json
```

## パフォーマンス指標

### パープレキシティ (Perplexity)

- **良好**: < 20
- **普通**: 20-50
- **要改善**: > 50

### Top-1精度 (Accuracy)

- **優秀**: > 0.4
- **良好**: 0.2-0.4
- **要改善**: < 0.2

### モデルサイズ別期待値

| モデルサイズ | パラメータ数 | 期待パープレキシティ | 期待精度  |
| ------------ | ------------ | -------------------- | --------- |
| GPT2-small   | 124M         | 15-25                | 0.25-0.35 |
| GPT2-medium  | 350M         | 12-20                | 0.30-0.40 |
| GPT2-large   | 774M         | 10-18                | 0.35-0.45 |
| GPT2-xl      | 1.5B         | 8-15                 | 0.40-0.50 |

## トラブルシューティング

### よくあるエラーと解決方法

#### 1. チェックポイント読み込みエラー

```text
Error: checkpoint loading failed
```

**解決方法**: チェックポイントファイルのパスと形式を確認してください。

#### 2. トークナイザーエラー

```text
Error: tokenizer initialization failed
```

**解決方法**:

- ドメインパラメータが正しく指定されているか確認
- 語彙ファイルのパスが正しいか確認
- 必要な依存関係がインストールされているか確認

#### 3. メモリ不足エラー

```text
CUDA out of memory
```

**解決方法**:

- `--max_test_samples` を減らす
- バッチサイズを調整する
- CPUモードを使用: `--device=cpu`

#### 4. データセット読み込みエラー

```text
Dataset loading failed
```

**解決方法**:

- データセットディレクトリのパスを確認
- データセットパラメータのJSON形式を確認

### パフォーマンス最適化

#### GPU使用量の最適化

```bash
# メモリ使用量を減らす
python gpt2/test_checkpoint.py \
    --checkpoint_path=ckpt.pt \
    --max_test_samples=100 \
    --device=cuda:0

# CPUで実行
python gpt2/test_checkpoint.py \
    --checkpoint_path=ckpt.pt \
    --device=cpu
```

## 設定ファイル

テスト設定は `gpt2/test_configs/` ディレクトリの設定ファイルで管理できます:

- `compounds_test_config.py`
- `molecule_nat_lang_test_config.py`
- `genome_test_config.py`
- `protein_sequence_test_config.py`
- `rna_test_config.py`

## 依存関係

- PyTorch >= 1.9.0
- Transformers >= 4.20.0
- NumPy
- tqdm
- matplotlib

## 注意事項

1. **GPU メモリ**: 大きなモデルのテストにはGPUメモリが必要です
2. **データセット**: 正確な評価には実際のテストデータセットが必要です
3. **トークナイザー**: ドメイン特化トークナイザーは対応するモジュールが必要です
4. **時間**: 大規模モデルのテストには時間がかかる場合があります

## 参考資料

- [GPT2論文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
