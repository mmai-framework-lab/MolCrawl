# ChemBERTa-2 Training Guide

## Overview

ChemBERTa-2は、SMILES化合物データに特化したRoBERTaベースのTransformerモデルです。大規模な化合物データベース（Organix13など）で事前学習し、分子特性予測や化合物生成タスクへの転移学習を可能にします。

## Features

### 🧪 SMILES Compounds特化

- SMILES専用のトークナイゼーション
- 化合物特性予測への高い転移学習性能
- 分子構造の深い理解

### 🔧 Technical Specifications

| Model Size | Parameters | Hidden Size | Layers | Attention Heads | Intermediate Size |
| ---------- | ---------- | ----------- | ------ | --------------- | ----------------- |
| Small      | ~10M       | 384         | 6      | 6               | 1536              |
| Medium     | ~85M       | 768         | 12     | 12              | 3072              |
| Large      | ~355M      | 1024        | 24     | 16              | 4096              |

### ⚙️ Training Configuration

- **Architecture**: RoBERTa (BERTの改良版)
- **Learning Rate**: 6e-5 (SMILES化合物に最適化)
- **Batch Size**: 128 per device
- **Gradient Accumulation**: 1 step (effective batch size = 128)
- **Max Sequence Length**: 256 tokens
- **Mixed Precision**: FP16
- **Optimizer**: AdamW with linear learning rate schedule
- **Warmup Steps**: 10,000

## Quick Start

### 1. データセットの確認

```bash
# Organix13化合物データセットの場所を確認
ls -la learning_source
```

### 2. トレーニングの実行

#### Small Model

```bash
CUDA_VISIBLE_DEVICES=0 ./workflows/03g-compounds-train-chemberta2-small.sh
```

#### Medium Model

```bash
CUDA_VISIBLE_DEVICES=0,1 ./workflows/03g-compounds-train-chemberta2-medium.sh
```

#### Large Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./workflows/03g-compounds-train-chemberta2-large.sh
```

### 3. Weights & Biases Integration

```bash
# Weights & Biasesを有効化してトレーニング
LEARNING_SOURCE_DIR=learning_source \
USE_WANDB=True \
WANDB_PROJECT=chemberta2-compounds \
  ./workflows/03g-compounds-train-chemberta2-small.sh
```

## Dataset Format

ChemBERTa-2は、SMILES形式の化合物データを使用します：

- **Input**: SMILES文字列（例: `CC(C)Cc1ccc(cc1)C(C)C(=O)O`）
- **Vocabulary**: 612トークン（SMILES文字と特殊トークン）
- **Sequence Length**: 256トークン（典型的なSMILES長: 50-150）
- **Dataset Size**: ~10M compounds (Organix13)

## Model Architecture

ChemBERTa-2は、RoBERTaアーキテクチャに基づいています：

1. **Input Embedding**: SMILESトークンをベクトル表現に変換
2. **RoBERTa Encoder**: 複数層のself-attention機構（BERTより改良）
3. **Masked Language Modeling**: ランダムにマスクされたトークンを予測

### RoBERTa vs BERT

- **Dynamic Masking**: 毎エポックで異なるマスキングパターン
- **No NSP**: Next Sentence Prediction タスクなし
- **Larger Batches**: より大きなバッチサイズ
- **More Data**: より大規模なデータで学習

## Directory Structure

```text
chemberta2/
├── main.py                      # メイン学習スクリプト
├── configurator.py              # 設定ファイルローダー
└── configs/
    └── compounds.py             # SMILES化合物設定

workflows/
├── 03g-compounds-train-chemberta2-small.sh
├── 03g-compounds-train-chemberta2-medium.sh
└── 03g-compounds-train-chemberta2-large.sh

learning_source
└── compounds/
    ├── organix13/
    │   └── compounds/
    │       └── training_ready_hf_dataset/  # 学習データ
    └── chemberta2-output/       # モデル出力
```

## Training Process

### 1. データの読み込み

```python
from datasets import load_from_disk

train_dataset = load_from_disk("learning_source")
```

### 2. トークナイゼーション

SMILES専用の語彙を使用：

- `[PAD]`: パディングトークン
- `[CLS]`: 分類トークン（文の開始）
- `[SEP]`: セパレータトークン（文の終了）
- `[MASK]`: マスクされたトークン
- SMILES文字: `C`, `N`, `O`, `=`, `(`, `)`, etc.

### 3. モデルの学習

Masked Language Modeling (MLM) タスクで学習：

- 15%のトークンをランダムにマスク
- マスクされたトークンを予測
- Cross-entropy loss

## Monitoring

### Weights & Biases Metrics

- `train/loss`: 学習損失
- `eval/loss`: 検証損失
- `train/learning_rate`: 学習率
- `train/epoch`: エポック数

### Local Logs

ログファイルは以下の場所に保存されます：

```text
learning_source{size}-{timestamp}.log
```

## Troubleshooting

### メモリ不足 (OOM)

```bash
# バッチサイズを削減
python chemberta2/main.py --config chemberta2/configs/compounds.py --batch_size 64
```

### 学習が遅い

```bash
# Gradient accumulationを調整（バッチサイズを減らした場合）
python chemberta2/main.py \
  --config chemberta2/configs/compounds.py \
  --batch_size 64 \
  --gradient_accumulation_steps 2
```

### データセットが見つからない

```bash
# LEARNING_SOURCE_DIRを設定
export LEARNING_SOURCE_DIR=learning_source
./workflows/03g-compounds-train-chemberta2-small.sh
```

## Performance Benchmarks

| Model Size | GPU Memory | Training Speed | Time to 300K steps |
| ---------- | ---------- | -------------- | ------------------ |
| Small      | ~6 GB      | ~5,000 steps/h | ~60 hours          |
| Medium     | ~16 GB     | ~2,500 steps/h | ~120 hours         |
| Large      | ~40 GB     | ~1,000 steps/h | ~300 hours         |

\*Benchmarks on NVIDIA A100 40GB GPU

## Advanced Usage

### Custom Configuration

設定ファイルを編集して、カスタマイズ可能：

```python
# chemberta2/configs/compounds.py
learning_rate = 1e-4  # 学習率の変更
max_steps = 500000    # ステップ数の変更
batch_size = 256      # バッチサイズの変更
```

### Resume Training

チェックポイントから自動的に再開：

```bash
# 同じコマンドを実行するだけで、最新のチェックポイントから再開
./workflows/03g-compounds-train-chemberta2-small.sh
```

### Fine-tuning for Downstream Tasks

事前学習済みモデルを使用して、特定のタスクにFine-tuning：

```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Load pre-trained model
model = RobertaForSequenceClassification.from_pretrained(
    "learning_source",
    num_labels=2  # Binary classification example
)

# Fine-tune on your task
# ... training code ...
```

## Comparison with Other Models

| Feature       | ChemBERTa-2      | BERT         | RoBERTa      | MolFormer   |
| ------------- | ---------------- | ------------ | ------------ | ----------- |
| Domain        | SMILES compounds | General text | General text | Molecules   |
| Architecture  | RoBERTa          | BERT         | RoBERTa      | Transformer |
| Tokenization  | SMILES chars     | WordPiece    | BPE          | SMILES      |
| Max Length    | 256              | 512          | 512          | 512         |
| Learning Rate | 6e-5             | 1e-4         | 6e-4         | 5e-5        |
| Year          | 2021/2026        | 2018         | 2019         | 2022        |

## Use Cases

1. **Molecular Property Prediction**: 毒性、溶解度、活性予測
2. **Drug Discovery**: リード化合物の最適化
3. **Retrosynthesis**: 合成経路の予測
4. **Molecule Generation**: 新規化合物の生成
5. **Reaction Prediction**: 化学反応の予測

## Citation

If you use ChemBERTa-2 in your research, please cite:

```bibtex
@article{chemberta2021,
  title={ChemBERTa-2: Towards Chemical Foundation Models},
  author={Ahmad et al.},
  journal={arXiv preprint},
  year={2021}
}
```

## FAQ

**Q: ChemBERTa-2とChemBERTaの違いは？**
A: ChemBERTa-2は、より大規模なデータと改良されたハイパーパラメータで学習されています。

**Q: 他のSMILESデータで学習できますか？**
A: はい、任意のSMILESデータセットで学習できます。トークナイザーは同じものを使用してください。

**Q: Fine-tuningの推奨設定は？**
A: Learning rate: 1e-5～5e-5、Batch size: 32～64、Epochs: 3～10を推奨します。

**Q: どのサイズのモデルを選ぶべきですか？**
A: Smallは開発・テスト用、Mediumは標準的なタスク用、Largeは高精度が必要なタスク用です。

## Support
