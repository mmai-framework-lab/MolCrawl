# ChemBERTa-2 実装サマリー

## 作成ファイル

### コア実装

1. **chemberta2/main.py**（370行）
   - `CompoundsDatasetLoader` クラスを含むメイン学習スクリプト
   - RoBERTa ベースのアーキテクチャ
   - チェックポイント再開対応
   - Weights & Biases 連携
   - 混合精度学習（FP16）

2. **chemberta2/configurator.py**（53行）
   - コマンドライン引数のパース
   - 設定ファイルの読み込みと上書き対応

3. **chemberta2/configs/compounds.py**（115行）
   - SMILES 化合物データセット用設定
   - SMILES トークナイザー設定（612トークン）
   - 前処理関数
   - Organix13 データセット統合

### ブートストラップスクリプト

1. **workflows/03g-compounds-train-chemberta2-small.sh**
2. **workflows/03g-compounds-train-chemberta2-medium.sh**
3. **workflows/03g-compounds-train-chemberta2-large.sh**
   - 3つのモデルサイズ用学習スクリプト
   - 環境変数設定
   - 自動ログ出力

### ドキュメント

1. **docs/CHEMBERTA2_TRAINING_GUIDE.md**
   - 包括的ユーザーガイド
   - クイックスタート例
   - トラブルシューティング
   - 性能ベンチマーク

## クイックスタート

### 学習コマンド

```bash
# Small model（テスト推奨）
CUDA_VISIBLE_DEVICES=0 ./workflows/03g-compounds-train-chemberta2-small.sh

# Medium model
CUDA_VISIBLE_DEVICES=0 ./workflows/03g-compounds-train-chemberta2-medium.sh

# Large model
CUDA_VISIBLE_DEVICES=0,1,2,3 ./workflows/03g-compounds-train-chemberta2-large.sh

# Weights & Biases 有効化
LEARNING_SOURCE_DIR=learning_source \
USE_WANDB=True \
WANDB_PROJECT=chemberta2-compounds \
  ./workflows/03g-compounds-train-chemberta2-small.sh
```

## モデル仕様

| Model Size | Parameters | Hidden | Layers | Heads | Intermediate |
| ---------- | ---------- | ------ | ------ | ----- | ------------ |
| Small      | ~10M       | 384    | 6      | 6     | 1536         |
| Medium     | ~85M       | 768    | 12     | 12    | 3072         |
| Large      | ~355M      | 1024   | 24     | 16    | 4096         |

## 主要機能

### SMILES 化合物特化

- **トークナイズ**: SMILES 文字ベース（612トークン）
- **最大長**: 256 トークン（SMILES に最適）
- **アーキテクチャ**: RoBERTa（BERT 改良版）
- **データセット**: Organix13（約1,000万化合物）

### 学習最適化

- **学習率**: 6e-5（SMILES 向け最適化）
- **バッチサイズ**: device ごとに 128
- **Gradient Accumulation**: 1（実効バッチ=128）
- **混合精度**: FP16 で高速化
- **Warmup**: 線形スケジュールで 10,000 steps

### RoBERTa の BERT に対する改善点

- 動的マスキング（エポックごとに異なる）
- Next Sentence Prediction（NSP）なし
- より大きなバッチと大量データで学習
- 下流タスク性能の改善

## 他実装との比較

| Feature       | ChemBERTa-2      | DNABERT-2     | ESM-2             | RNAformer         |
| ------------- | ---------------- | ------------- | ----------------- | ----------------- |
| Domain        | SMILES compounds | DNA sequences | Protein sequences | RNA transcriptome |
| Architecture  | RoBERTa          | BERT          | ESM (BERT-like)   | BERT              |
| Tokenization  | SMILES chars     | BPE           | Amino acids       | Gene IDs          |
| Vocab Size    | 612              | ~4K           | ~33               | ~60K              |
| Max Length    | 256              | 1024          | 1024              | 1024              |
| Learning Rate | 6e-5             | 3e-5          | 4e-4              | 1e-4              |
| Batch Size    | 128              | 16            | 4                 | 8                 |
| Dropout       | 0.1              | 0.1           | 0.0               | 0.1               |

## 主なユースケース

1. **分子特性予測**: 毒性、溶解性、活性予測
2. **創薬**: リード最適化、ADMET 予測
3. **逆合成**: 合成経路予測
4. **分子生成**: 新規化合物設計
5. **反応予測**: 化学反応の生成物予測

## 想定性能

### 学習メトリクス

- **Loss**: 300K steps 後に ~1.5-2.0 へ低下が目安
- **Perplexity**: 良好な収束で ~5-8 が目標
- **メモリ使用量**: モデルサイズにより 6-40GB

### 計算要件

- **Small**: A100 で約60時間（300K steps）
- **Medium**: A100 で約120時間（300K steps）
- **Large**: A100 で約300時間（300K steps）

## 技術詳細

### データフロー

1. SMILES 語彙（612トークン）を読み込み
2. Organix13 化合物データセットを読み込み
3. Attention mask を付与
4. 15% マスキングで MLM を適用
5. AdamW で学習

### ディレクトリ構造

```text
learning_source
└── compounds/
    ├── organix13/
    │   └── compounds/
    │       └── training_ready_hf_dataset/  # ~10M compounds
    │           ├── train/
    │           ├── valid/
    │           └── test/
    ├── chemberta2-output/            # Checkpoints
    │   ├── chemberta2-small/
    │   ├── chemberta2-medium/
    │   └── chemberta2-large/
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
python -c "with open('assets/molecules/vocab.txt') as f: \
  print(f'Vocab size: {len(f.readlines())}')"

# 3. トークナイザー確認
python -c "from compounds.utils.tokenizer import CompoundsTokenizer; \
  tok = CompoundsTokenizer('assets/molecules/vocab.txt', 256); \
  print(f'Tokenizer loaded: {len(tok)} tokens')"
```

### 学習後チェック

```bash
# モデル出力確認
ls -la learning_source

# ログ確認
tail -f learning_source
```

## よくある問題

### 問題: 語彙ファイルが見つからない

```bash
# 解決: assets 配下を確認
ls -la assets/molecules/vocab.txt
```

### 問題: OOM（メモリ不足）

```bash
# 解決: バッチサイズを下げる
python chemberta2/main.py --config chemberta2/configs/compounds.py --batch_size 64
```

### 問題: 学習が遅い

```bash
# 解決: バッチサイズを下げた場合は accumulation で補う
python chemberta2/main.py --batch_size 64 --gradient_accumulation_steps 2
```

## 参考文献

- **ChemBERTa 論文**: "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction" (arXiv, 2020)
- **RoBERTa 論文**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (arXiv, 2019)
- **Organix13**: 大規模有機化合物データベース
- **Hugging Face Transformers**: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)

## 次のステップ

1. **学習開始**: まず small モデルでセットアップを検証
2. **進捗監視**: Weights & Biases ダッシュボードで確認
