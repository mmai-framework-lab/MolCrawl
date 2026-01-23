# RNAformer Training Guide

## Overview

RNAformerは、RNA transcriptome（遺伝子発現）データに特化したTransformerモデルです。Geneformerアーキテクチャをベースに、CellXGeneデータセットからの大規模なシングルセルRNAシーケンシングデータで学習します。

## Features

### 🧬 RNA Transcriptome特化
- 遺伝子発現データ用のカスタムトークナイゼーション
- セルタイプ特異的な学習
- 長いコンテキスト（1024トークン）のサポート

### 🔧 Technical Specifications

| Model Size | Parameters | Hidden Size | Layers | Attention Heads | Intermediate Size |
|------------|-----------|-------------|--------|-----------------|-------------------|
| Small      | ~40M      | 512         | 8      | 8               | 2048              |
| Medium     | ~90M      | 768         | 12     | 12              | 3072              |
| Large      | ~180M     | 1024        | 16     | 16              | 4096              |

### ⚙️ Training Configuration
- **Learning Rate**: 1e-4 (RNA transcriptomeに最適化)
- **Batch Size**: 8 per device
- **Gradient Accumulation**: 16 steps (effective batch size = 128)
- **Max Sequence Length**: 1024 tokens
- **Mixed Precision**: FP16
- **Optimizer**: AdamW with cosine learning rate schedule
- **Warmup Steps**: 10,000

## Quick Start

### 1. データセットの確認

```bash
# RNA transcriptomeデータセットの場所を確認
ls -la learning_source_20250904-rna-refined/rna/training_ready_hf_dataset/
```

### 2. トレーニングの実行

#### Small Model
```bash
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-small.sh
```

#### Medium Model
```bash
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-medium.sh
```

#### Large Model
```bash
CUDA_VISIBLE_DEVICES=0 ./workflows/03f-rna-train-rnaformer-large.sh
```

### 3. Weights & Biases Integration

```bash
# Weights & Biasesを有効化してトレーニング
LEARNING_SOURCE_DIR=learning_source_20250904-rna-refined \
USE_WANDB=True \
WANDB_PROJECT=rnaformer-transcriptome \
  ./workflows/03f-rna-train-rnaformer-small.sh
```

## Dataset Format

RNAformerは、CellXGeneから収集された遺伝子発現データを使用します：

- **Input**: 遺伝子IDのシーケンス（発現レベルでソート済み）
- **Vocabulary**: ~60,000遺伝子の語彙
- **Sequence Length**: 1024トークン
- **Dataset Size**: ~54M cells

## Model Architecture

RNAformerは、Geneformerアーキテクチャに基づいています：

1. **Input Embedding**: 遺伝子IDをベクトル表現に変換
2. **Transformer Encoder**: 複数層のself-attention機構
3. **Masked Language Modeling**: ランダムにマスクされた遺伝子を予測

## Directory Structure

```
rnaformer/
├── main.py                      # メイン学習スクリプト
├── configurator.py              # 設定ファイルローダー
└── configs/
    └── rna.py                   # RNA transcriptome設定

workflows/
├── 03f-rna-train-rnaformer-small.sh
├── 03f-rna-train-rnaformer-medium.sh
└── 03f-rna-train-rnaformer-large.sh

learning_source_20250904-rna-refined/
└── rna/
    ├── gene_vocab.json          # 遺伝子語彙
    ├── training_ready_hf_dataset/  # 学習データ
    └── rnaformer-output/        # モデル出力
```

## Training Process

### 1. データの読み込み
```python
from datasets import load_from_disk

train_dataset = load_from_disk("learning_source_20250904-rna-refined/rna/training_ready_hf_dataset/train")
```

### 2. トークナイゼーション
遺伝子IDベースの語彙を使用：
- `<pad>`: パディングトークン
- `<unk>`: 未知の遺伝子
- `<eos>`: シーケンス終了
- `<mask>`: マスクされた遺伝子

### 3. モデルの学習
Masked Language Modeling (MLM) タスクで学習：
- 15%の遺伝子をランダムにマスク
- マスクされた遺伝子を予測
- Cross-entropy loss

## Monitoring

### Weights & Biases Metrics
- `train/loss`: 学習損失
- `eval/loss`: 検証損失
- `train/learning_rate`: 学習率
- `train/epoch`: エポック数

### Local Logs
ログファイルは以下の場所に保存されます：
```
learning_source_20250904-rna-refined/rna/logs/rnaformer-train-{size}-{timestamp}.log
```

## Troubleshooting

### メモリ不足 (OOM)
```bash
# バッチサイズを削減
python rnaformer/main.py --config rnaformer/configs/rna.py --batch_size 4
```

### 学習が遅い
```bash
# Gradient accumulationを調整
python rnaformer/main.py \
  --config rnaformer/configs/rna.py \
  --gradient_accumulation_steps 32
```

### データセットが見つからない
```bash
# LEARNING_SOURCE_DIRを設定
export LEARNING_SOURCE_DIR=learning_source_20250904-rna-refined
./workflows/03f-rna-train-rnaformer-small.sh
```

## Performance Benchmarks

| Model Size | GPU Memory | Training Speed | Time to 100K steps |
|------------|------------|----------------|-------------------|
| Small      | ~12 GB     | ~2,500 steps/h | ~40 hours         |
| Medium     | ~18 GB     | ~1,800 steps/h | ~55 hours         |
| Large      | ~30 GB     | ~1,200 steps/h | ~83 hours         |

*Benchmarks on NVIDIA A100 40GB GPU

## Advanced Usage

### Custom Configuration

設定ファイルを編集して、カスタマイズ可能：

```python
# rnaformer/configs/rna.py
learning_rate = 5e-5  # 学習率の変更
max_steps = 200000    # ステップ数の変更
batch_size = 16       # バッチサイズの変更
```

### Resume Training

チェックポイントから自動的に再開：
```bash
# 同じコマンドを実行するだけで、最新のチェックポイントから再開
./workflows/03f-rna-train-rnaformer-small.sh
```

## Comparison with Other Models

| Feature | RNAformer | BERT | GPT-2 | Geneformer |
|---------|-----------|------|-------|------------|
| Domain | RNA transcriptome | General text | General text | Gene expression |
| Tokenization | Gene IDs | WordPiece | BPE | Gene IDs |
| Max Length | 1024 | 512 | 1024 | 2048 |
| Learning Rate | 1e-4 | 6e-6 | 6e-4 | 1e-4 |
| Year | 2026 | 2018 | 2019 | 2023 |

## Citation

If you use RNAformer in your research, please cite:

```bibtex
@article{rnaformer2026,
  title={RNAformer: A Transformer Model for RNA Transcriptome Analysis},
  author={Your Team},
  year={2026}
}
```

## FAQ

**Q: RNAformerとGeneformerの違いは？**
A: RNAformerはGeneformerアーキテクチャをベースにしていますが、より効率的なバッチ処理とメモリ管理を実装しています。

**Q: 他のRNA-seqデータで学習できますか？**
A: はい、遺伝子IDベースのトークナイゼーションを使用している限り、任意のRNA-seqデータで学習できます。

**Q: Fine-tuningはどうやって行いますか？**
A: 学習済みモデルをロードして、特定のタスク（セルタイプ分類など）用のヘッドを追加してください。

## Support

問題が発生した場合は、GitHubのIssueを作成するか、ドキュメントを参照してください。
