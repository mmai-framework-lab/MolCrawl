# ESM-2 Training Guide for Protein Sequence Data

## 概要

このドキュメントでは、既存のprotein_sequenceデータセット（UniProt）を使用してESM-2モデルを学習する方法を説明します。

## ESM-2とは

ESM-2 (Evolutionary Scale Modeling 2) は、Metaが開発したタンパク質配列専用の最先端トランスフォーマーモデルです。

### 主な特徴

1. **進化的スケールでの学習**
   - 数億のタンパク質配列で事前学習
   - 進化的な関係性を捉えた表現学習

2. **スケーラブルなアーキテクチャ**
   - 8M から 15B パラメータまでスケール可能
   - 効率的な学習と推論

3. **幅広いタスクに対応**
   - Structure prediction (AlphaFold2レベル)
   - Function annotation
   - Variant effect prediction
   - Contact prediction

### 既存BERTとの比較

| 項目                 | BERT (既存) | ESM-2          |
| -------------------- | ----------- | -------------- |
| ドメイン             | 汎用        | タンパク質専用 |
| トークナイゼーション | 文字レベル  | アミノ酸レベル |
| 学習率               | 6e-6        | 4e-4           |
| Dropout              | 0.1         | 0.0            |
| Position embeddings  | Learned     | Learned        |
| 最適化               | AdamW       | AdamW          |
| 収束速度             | 普通        | 高速           |

## セットアップ

### 1. 必要な依存関係

既存の環境に追加のパッケージは不要です。transformers、datasetsが既にインストールされています。

### 2. ディレクトリ構造

```text
esm2/
├── main.py                     # メイン学習スクリプト
├── configurator.py             # 設定ローダー
└── configs/
    └── protein_sequence.py     # protein_sequence用の設定

workflows/
├── 03e-protein_sequence-train-esm2-small.sh
├── 03e-protein_sequence-train-esm2-medium.sh
└── 03e-protein_sequence-train-esm2-large.sh
```

## 使用方法

### 基本的な学習実行

#### Small モデル (推奨: 開発・テスト用)

```bash
# 単一GPU
CUDA_VISIBLE_DEVICES=0 ./workflows/03e-protein_sequence-train-esm2-small.sh

# Weights & Biases ログ有効化
CUDA_VISIBLE_DEVICES=0 USE_WANDB=True WANDB_PROJECT=esm2-protein \
  ./workflows/03e-protein_sequence-train-esm2-small.sh
```

#### Medium モデル (推奨: 実験用)

```bash
# 2 GPUs推奨
CUDA_VISIBLE_DEVICES=0,1 ./workflows/03e-protein_sequence-train-esm2-medium.sh
```

#### Large モデル (推奨: 本番用)

```bash
# 4 GPUs推奨 (A100 40GB以上)
CUDA_VISIBLE_DEVICES=0,1,2,3 ./workflows/03e-protein_sequence-train-esm2-large.sh
```

### ログの確認

```bash
# リアルタイムでログを確認
tail -f $LEARNING_SOURCE_DIR/protein_sequence/logs/esm2-train-small-*.log

# 最新のログファイルを表示
ls -lt $LEARNING_SOURCE_DIR/protein_sequence/logs/esm2-train-*.log | head -1
```

### 学習の停止

```bash
# プロセスIDを確認
ps aux | grep esm2

# 停止
kill <PID>
```

## モデルサイズの選択

### Small (~8M パラメータ)

- **Hidden size**: 320
- **Layers**: 6
- **Attention heads**: 20
- **GPU メモリ**: ~8GB (バッチサイズ4)
- **用途**: 開発、テスト、プロトタイピング
- **学習時間**: ~3-5日 (単一GPU)

### Medium (~35M パラメータ)

- **Hidden size**: 480
- **Layers**: 12
- **Attention heads**: 20
- **GPU メモリ**: ~16GB (バッチサイズ4)
- **用途**: 実験、中規模タスク
- **学習時間**: ~5-7日 (2 GPUs)

### Large (~150M パラメータ)

- **Hidden size**: 640
- **Layers**: 30
- **Attention heads**: 20
- **GPU メモリ**: ~24GB (バッチサイズ4)
- **用途**: 本番、最高精度が必要なタスク
- **学習時間**: ~7-10日 (4 GPUs)

## 設定のカスタマイズ

### コマンドラインオプション

```bash
python esm2/main.py esm2/configs/protein_sequence.py \
  --model_size=medium \
  --max_steps=600000 \
  --learning_rate=5e-4 \
  --batch_size=8 \
  --save_steps=10000 \
  --use_wandb=True \
  --wandb_project=my-esm2-project
```

### 設定ファイルの編集

[esm2/configs/protein_sequence.py](../../molcrawl/esm2/configs/protein_sequence.py) を編集:

```python
# 学習ステップ数を増やす
max_steps = 600000

# より大きいバッチサイズ（GPU メモリに余裕がある場合）
batch_size = 8
gradient_accumulation_steps = 16  # Effective: 8 * 16 = 128

# より長い配列に対応（必要な場合）
max_length = 2048  # デフォルト: 1024

# 保存頻度を変更
save_steps = 10000  # デフォルト: 5000
```

## データセット

### 使用するデータセット

既存の `protein_sequence` データセット（UniProt UniRef50）を使用します:

```text
$LEARNING_SOURCE_DIR/protein_sequence/training_ready_hf_dataset/
```

### データセットの特徴

- **ソース**: UniProt UniRef50
- **内容**: タンパク質配列（アミノ酸配列）
- **トークナイザー**: ESM character-level tokenizer
- **フォーマット**: HuggingFace Datasets (Arrow形式)

## トラブルシューティング

### GPU メモリ不足

```bash
# バッチサイズを減らす
python esm2/main.py esm2/configs/protein_sequence.py --batch_size=2

# または config ファイルで gradient_accumulation_steps を増やす
# Effective batch sizeは維持される
```

### トークナイザーエラー

ESM-2は既存のESMトークナイザー（BERT互換ラッパー）を使用します:

```bash
# トークナイザーの確認
python -c "from protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer; t = create_bert_protein_tokenizer(); print(len(t.get_vocab()))"
```

### チェックポイントから再開

学習は自動的に最新のチェックポイントから再開されます:

```bash
# チェックポイントディレクトリを確認
ls -lt $LEARNING_SOURCE_DIR/protein_sequence/esm2-output/esm2-small/
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

### 3. Gradient Accumulation

タンパク質配列は長いため、小さいバッチサイズとgradient accumulationを組み合わせます:

```python
batch_size = 4
gradient_accumulation_steps = 32
# Effective batch size = 128
```

## 評価とファインチューニング

### ダウンストリームタスクへの適用

学習済みモデルは以下のタスクに使用できます:

1. **ProteinGym評価** (変異効果予測)

   ```bash
   ./workflows/run_bert_proteingym_evaluation.sh \
     --model_path $LEARNING_SOURCE_DIR/protein_sequence/esm2-output/esm2-small/checkpoint-100000
   ```

2. **Structure prediction**
3. **Function annotation**
4. **Contact prediction**

## ESM-2の論文・参考文献

- [Language models of protein sequences at the scale of evolution enable accurate structure prediction](https://www.science.org/doi/10.1126/science.ade2574)
- [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.pnas.org/doi/10.1073/pnas.2016239118)
- [ESM GitHub Repository](https://github.com/facebookresearch/esm)

## 技術的な詳細

### アーキテクチャの違い

**BERT (既存)**:

- 汎用的なトランスフォーマー
- 文字レベルトークナイゼーション
- Standard attention

**ESM-2**:

- タンパク質専用に最適化
- アミノ酸レベルトークナイゼーション
- Attention bias for protein structure
- No dropout (0.0)

### ハイパーパラメータの選択

学習率の決定:

- Small: 4e-4 (推奨)
- Medium: 3e-4
- Large: 2e-4

ウォームアップステップ:

- Default: 2000
- 大規模データセット: 5000

## よくある質問 (FAQ)

**Q: BERTとESM-2の両方を学習すべきですか？**

A: ESM-2の方が新しく、タンパク質配列に特化しているため、通常はESM-2を推奨します。

**Q: 既存のBERTモデルからESM-2に移行できますか？**

A: 直接的な重みの転送は難しいですが、同じデータセットで再学習することで高い性能が得られます。

**Q: データセットを新しく作成する必要がありますか？**

A: いいえ。既存の `protein_sequence` データセットをそのまま使用できます。

**Q: どのモデルサイズを選べばいいですか？**

A:

- 開発・テスト: Small
- 実験・研究: Medium
- 本番・最高精度: Large

**Q: 学習にどれくらい時間がかかりますか？**

A: Small モデルで3-5日（単一GPU）、Large モデルで7-10日（4 GPUs）が目安です。

## サポート

問題が発生した場合は、以下を確認してください:

1. ログファイル: `$LEARNING_SOURCE_DIR/protein_sequence/logs/`
2. GPU メモリ使用量: `nvidia-smi`
3. データセットの存在: `ls $LEARNING_SOURCE_DIR/protein_sequence/training_ready_hf_dataset/`

---

**作成日**: 2026-01-22
**バージョン**: 1.0.0
**対応モデル**: ESM-2 (Small, Medium, Large)
