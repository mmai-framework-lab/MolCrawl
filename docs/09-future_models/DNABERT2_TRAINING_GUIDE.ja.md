# DNABERT-2 Training Guide for Genome Sequence Data

## 概要

このドキュメントでは、既存のgenome_sequenceデータセットを使用してDNABERT-2モデルを学習する方法を説明します。

## DNABERT-2とは

DNABERT-2は、DNA配列解析に特化した最先端のBERTベースモデルです。

### 主な特徴

1. **BPE (Byte Pair Encoding) トークナイゼーション**
   - 従来のk-merベースではなくBPEを使用
   - より効率的で柔軟なトークン化

2. **最適化されたアーキテクチャ**
   - DNA配列の特性を考慮した設計
   - より少ないパラメータで高性能

3. **効率的な学習**
   - 既存のBERTより高速に収束
   - より少ない計算リソースで高精度

### 既存BERTとの比較

| 項目                 | BERT (既存) | DNABERT-2    |
| -------------------- | ----------- | ------------ |
| トークナイゼーション | k-mer       | BPE          |
| 最大長               | 1024        | 512 (効率的) |
| 学習率               | 6e-6        | 3e-5         |
| バッチサイズ         | 8           | 16           |
| MLM確率              | 0.2         | 0.15         |
| 収束速度             | 遅い        | 速い         |

## セットアップ

### 1. 必要な依存関係

既存の環境に追加のパッケージは不要です。transformers、datasets、sentencepieceが既にインストールされています。

### 2. ディレクトリ構造

```text
dnabert2/
├── main.py                     # メイン学習スクリプト
├── configurator.py             # 設定ローダー
└── configs/
    └── genome_sequence.py      # genome_sequence用の設定

workflows/
├── 03d-genome_sequence-train-dnabert2-small.sh
├── 03d-genome_sequence-train-dnabert2-medium.sh
└── 03d-genome_sequence-train-dnabert2-large.sh
```

## 使用方法

### 基本的な学習実行

#### Small モデル (推奨: 開発・テスト用)

```bash
# 単一GPU
CUDA_VISIBLE_DEVICES=0 ./workflows/03d-genome_sequence-train-dnabert2-small.sh

# Weights & Biases ログ有効化
CUDA_VISIBLE_DEVICES=0 USE_WANDB=True WANDB_PROJECT=dnabert2-genome \
  ./workflows/03d-genome_sequence-train-dnabert2-small.sh
```

#### Medium モデル (推奨: 実験用)

```bash
# 2 GPUs推奨
CUDA_VISIBLE_DEVICES=0,1 ./workflows/03d-genome_sequence-train-dnabert2-medium.sh
```

#### Large モデル (推奨: 本番用)

```bash
# 4 GPUs推奨 (A100 40GB以上)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./workflows/03d-genome_sequence-train-dnabert2-large.sh
```

### ログの確認

```bash
# リアルタイムでログを確認
tail -f $LEARNING_SOURCE_DIR/genome_sequence/logs/dnabert2-train-small-*.log

# 最新のログファイルを表示
ls -lt $LEARNING_SOURCE_DIR/genome_sequence/logs/dnabert2-train-*.log | head -1
```

### 学習の停止

```bash
# プロセスIDを確認
ps aux | grep dnabert2

# 停止
kill <PID>
```

## モデルサイズの選択

### Small (768次元, 12層)

- **パラメータ数**: ~110M
- **GPU メモリ**: ~8GB (バッチサイズ16)
- **用途**: 開発、テスト、プロトタイピング
- **学習時間**: ~3-5日 (単一GPU)

### Medium (1024次元, 24層)

- **パラメータ数**: ~350M
- **GPU メモリ**: ~16GB (バッチサイズ16)
- **用途**: 実験、中規模タスク
- **学習時間**: ~5-7日 (2 GPUs)

### Large (1280次元, 32層)

- **パラメータ数**: ~600M
- **GPU メモリ**: ~24GB (バッチサイズ16)
- **用途**: 本番、最高精度が必要なタスク
- **学習時間**: ~7-10日 (4 GPUs)

## 設定のカスタマイズ

### コマンドラインオプション

```bash
python dnabert2/main.py dnabert2/configs/genome_sequence.py \
  --model_size=medium \
  --max_steps=300000 \
  --learning_rate=5e-5 \
  --batch_size=32 \
  --save_steps=10000 \
  --use_wandb=True \
  --wandb_project=my-dnabert2-project
```

### 設定ファイルの編集

[dnabert2/configs/genome_sequence.py](../../molcrawl/tasks/pretrain/configs/genome_sequence/dnabert2.py) を編集:

```python
# 学習ステップ数を増やす
max_steps = 300000

# より大きいバッチサイズ
batch_size = 32
gradient_accumulation_steps = 2  # Effective: 32 * 2 = 64

# より長い配列に対応
max_length = 1024  # デフォルト: 512

# 保存頻度を変更
save_steps = 10000  # デフォルト: 5000
```

## データセット

### 使用するデータセット

既存の `genome_sequence` データセット（RefSeq）を使用します:

```text
$LEARNING_SOURCE_DIR/genome_sequence/training_ready_hf_dataset/
```

### データセットの特徴

- **ソース**: NCBI RefSeq
- **生物種**: 複数種のゲノム配列
- **トークナイザー**: SentencePiece (既存)
- **フォーマット**: Hugging Face Datasets (Arrow形式)

### 独自データセットの使用

既存のデータセット準備パイプラインを使用して新しいデータセットを作成できます:

```bash
# genome_sequenceデータセット準備
./workflows/01-genome_sequence-prepare.sh
./workflows/02-genome_sequence-prepare-gpt2.sh
```

## トラブルシューティング

### GPU メモリ不足

```bash
# バッチサイズを減らす
python dnabert2/main.py dnabert2/configs/genome_sequence.py --batch_size=8

# Gradient accumulation を増やす（effective batch sizeは維持）
# config ファイルで gradient_accumulation_steps を調整
```

### トークナイザーエラー

```bash
# SentencePieceモデルが正しく配置されているか確認
ls -l $LEARNING_SOURCE_DIR/genome_sequence/spm_tokenizer.model

# 環境変数が設定されているか確認
echo $LEARNING_SOURCE_DIR
```

### チェックポイントから再開

学習は自動的に最新のチェックポイントから再開されます:

```bash
# 手動で再開
python dnabert2/main.py dnabert2/configs/genome_sequence.py
```

## パフォーマンス最適化

### 1. Mixed Precision Training

デフォルトで有効化されています (fp16=True):

```python
# training_args in main.py
fp16=True,  # Automatic mixed precision
```

### 2. データローディング並列化

```python
# training_args in main.py
dataloader_num_workers=4,  # 4 workers for data loading
```

### 3. Gradient Checkpointing (大モデル用)

メモリを節約する場合:

```python
# model_config に追加
model_config.gradient_checkpointing = True
```

## 評価とファインチューニング

### ダウンストリームタスクへの適用

学習済みモデルは以下のタスクに使用できます:

1. **ClinVar評価** (変異予測)

   ```bash
   ./workflows/run_bert_clinvar_evaluation.sh \
     --model-path $LEARNING_SOURCE_DIR/genome_sequence/dnabert2-output/dnabert2-small/checkpoint-100000
   ```

2. **配列分類** (fine-tuning)
3. **モチーフ検出**
4. **遺伝子発現予測**

## 参考文献

- [DNABERT-2 論文](https://arxiv.org/abs/2306.15006)
- [DNABERT-2 GitHub](https://github.com/MAGICS-LAB/DNABERT_2)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## 技術的な詳細

### アーキテクチャの違い

**BERT (既存)**:

- Position embeddings: Learned
- Token embeddings: k-mer based
- Attention: Standard multi-head

**DNABERT-2**:

- Position embeddings: Learned (optimized for DNA)
- Token embeddings: BPE-based
- Attention: Efficient attention mechanism

### ハイパーパラメータの選択

学習率の決定:

- Small: 3e-5 (推奨)
- Medium: 2e-5
- Large: 1e-5

ウォームアップステップ:

- Default: 10000
- 大規模データセット: 20000

## よくある質問 (FAQ)

**Q: BERTとDNABERT-2の両方を学習すべきですか？**

A: DNABERT-2の方が新しく、一般的に高性能です。新規プロジェクトではDNABERT-2を推奨します。

**Q: 既存のBERTモデルからDNABERT-2に移行できますか？**

A: 直接的な重みの転送は難しいですが、同じデータセットで再学習することで同等以上の性能が得られます。

**Q: データセットを新しく作成する必要がありますか？**

A: いいえ。既存の `genome_sequence` データセットをそのまま使用できます。

**Q: どのモデルサイズを選べばいいですか？**

A:

- 開発・テスト: Small
- 実験・研究: Medium
- 本番・最高精度: Large

**Q: 学習にどれくらい時間がかかりますか？**

A: Small モデルで3-5日（単一GPU）、Large モデルで7-10日（4 GPUs）が目安です。

## サポート

問題が発生した場合は、以下を確認してください:

1. ログファイル: `$LEARNING_SOURCE_DIR/genome_sequence/logs/`
2. GPU メモリ使用量: `nvidia-smi`
3. データセットの存在: `ls $LEARNING_SOURCE_DIR/genome_sequence/training_ready_hf_dataset/`

---

**作成日**: 2026-01-22
**バージョン**: 1.0.0
**対応モデル**: DNABERT-2 (Small, Medium, Large)
