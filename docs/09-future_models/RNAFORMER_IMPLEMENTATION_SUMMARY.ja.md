# RNAformer 実装サマリー

## ファイル作成一覧

### コア実装

1. **rnaformer/main.py** (348行)
   - `RNADatasetLoader` クラスを含むメイン学習スクリプト
   - チェックポイント再開に対応
   - Weights & Biases 連携
   - Mixed Precision Training（FP16）

2. **rnaformer/configurator.py** (53行)
   - コマンドライン引数の解析
   - オーバーライド付き設定ファイル読み込み

3. **rnaformer/configs/rna.py** (136行)
   - RNA トランスクリプトーム用データセット設定
   - 遺伝子語彙の読み込み
   - WordLevel トークナイザー設定
   - 前処理関数

### ブートストラップスクリプト

1. **workflows/03f-rna-train-rnaformer-small.sh**
2. **workflows/03f-rna-train-rnaformer-medium.sh**
3. **workflows/03f-rna-train-rnaformer-large.sh**
   - 3つのモデルサイズ向け実行可能学習スクリプト
   - 環境変数設定
   - 自動ログ出力

### ドキュメント

1. **docs/RNAFORMER_TRAINING_GUIDE.md**
   - 包括的ユーザーガイド
   - クイックスタート例
   - トラブルシューティング
   - 性能ベンチマーク

## クイックスタート

### 学習コマンド

```bash
# Small model (テスト用途に推奨)
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-small.sh

# Medium model
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-medium.sh

# Large model
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-large.sh

# Weights & Biases を有効化
LEARNING_SOURCE_DIR=learning_source \
USE_WANDB=True \
WANDB_PROJECT=rnaformer-transcriptome \
  ./workflows/03f-rna-train-rnaformer-small.sh
```

## モデル仕様

| Model Size | Parameters | Hidden | Layers | Heads | Intermediate |
| ---------- | ---------- | ------ | ------ | ----- | ------------ |
| Small      | ~40M       | 512    | 8      | 8     | 2048         |
| Medium     | ~90M       | 768    | 12     | 12    | 3072         |
| Large      | ~180M      | 1024   | 16     | 16    | 4096         |

## 主な機能

### RNA トランスクリプトーム特化

- **トークン化**: 遺伝子 ID ベース語彙（約60K遺伝子）
- **最大長**: 1024 トークン（細胞の発現プロファイル全体）
- **アーキテクチャ**: Geneformer ベース BERT エンコーダ
- **データセット**: CELLxGENE 単一細胞 RNA-seq データ（約5,400万細胞）

### 学習最適化

- **学習率**: 1e-4（RNA データ向け最適化値）
- **バッチサイズ**: デバイスあたり 8（メモリ効率重視）
- **Gradient Accumulation**: 16 ステップ（実効バッチ = 128）
- **Mixed Precision**: FP16 による高速化
- **Warmup**: 10,000 ステップ（cosine schedule）

### インフラ

- 自動チェックポイント再開
- Weights & Biases 連携
- 包括的ロギング
- メモリ効率の高いデータローディング

## 他実装との比較

| Feature       | RNAformer         | DNABERT-2     | ESM-2             |
| ------------- | ----------------- | ------------- | ----------------- |
| Domain        | RNA transcriptome | DNA sequences | Protein sequences |
| Tokenization  | Gene IDs          | BPE           | Amino acids       |
| Vocab Size    | ~60K              | ~4K           | ~33               |
| Max Length    | 1024              | 1024          | 1024              |
| Learning Rate | 1e-4              | 3e-5          | 4e-4              |
| Batch Size    | 8                 | 16            | 4                 |
| Dropout       | 0.1               | 0.1           | 0.0               |

## ユースケース

1. **細胞型分類**: 発現プロファイルから細胞型を予測
2. **遺伝子機能予測**: 共発現情報から遺伝子機能を推定
3. **疾患状態同定**: 健常細胞と疾患細胞の分類
4. **薬剤応答予測**: 処置に対する細胞応答を予測
5. **遺伝子制御ネットワーク**: 遺伝子間相互作用を学習

## 想定パフォーマンス

### 学習指標

- **Loss**: 100K ステップ後におよそ 2.5-3.0 まで低下
- **Perplexity**: 良好な収束で 12-20 を目標
- **メモリ使用量**: モデルサイズに応じて 12-30 GB

### 計算要件

- **Small**: A100 で約40時間（100K ステップ）
- **Medium**: A100 で約55時間（100K ステップ）
- **Large**: A100 で約83時間（100K ステップ）

## 技術詳細

### データフロー

1. 遺伝子語彙を読み込み（JSON形式）
2. WordLevel トークナイザーを作成
3. Hugging Face データセットを読み込み（Arrow形式）
4. attention mask を追加
5. 15% マスキングで MLM を適用
6. AdamW オプティマイザで学習

### ディレクトリ構造

```text
learning_source
└── rna/
    ├── gene_vocab.json              # ~60K gene IDs
    ├── training_ready_hf_dataset/   # Arrow format
    │   ├── train/
    │   └── test/
    ├── rnaformer-output/            # Checkpoints
    │   ├── rnaformer-small/
    │   ├── rnaformer-medium/
    │   └── rnaformer-large/
    └── logs/                        # Training logs
```

## 検証

### 事前チェック

```bash
# 1. データセット確認
python -c "from datasets import load_from_disk; \
  ds = load_from_disk('learning_source'); \
  print(f'Dataset size: {len(ds)}')"

# 2. 語彙確認
python -c "import json; \
  vocab = json.load(open('learning_source')); \
  print(f'Vocab size: {len(vocab)}')"

# 3. トークナイザー確認
python -c "from transformers import AutoTokenizer; \
  tok = AutoTokenizer.from_pretrained('custom_tokenizer_rnaformer'); \
  print(f'Tokenizer loaded: {len(tok)} tokens')"
```

### 事後チェック

```bash
# モデル出力確認
ls -la learning_source

# ログ確認
tail -f learning_source
```

## よくある問題

### 問題: 遺伝子語彙が見つからない

```bash
# 対策: LEARNING_SOURCE_DIR を確認
export LEARNING_SOURCE_DIR=learning_source
```

### 問題: OOM (Out of Memory)

```bash
# 対策: バッチサイズを下げる
python rnaformer/main.py --config rnaformer/configs/rna.py --batch_size 4
```

### 問題: 学習が遅い

```bash
# 対策: gradient accumulation を増やす
python rnaformer/main.py --gradient_accumulation_steps 32
```

## 参考文献

- **Geneformer 論文**: "Transfer learning enables predictions in network biology" (Nature, 2023)
- **CELLxGENE**: <https://cellxgene.cziscience.com/>
- **Hugging Face Transformers**: <https://huggingface.co/docs/transformers/>

## 次のステップ

1. **学習開始**: まず small model でセットアップ検証
2. **進捗監視**: Weights & Biases ダッシュボード確認
3. **評価**: 下流タスク（分類・クラスタリング）を実行
4. **ファインチューニング**: 用途別に適応
5. **スケールアップ**: より高性能な medium/large を学習
