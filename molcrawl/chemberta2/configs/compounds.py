"""
ChemBERTa-2 Configuration for SMILES Compounds Data

このconfigは既存のcompoundsデータセット（Organix13）を使用して
ChemBERTa-2モデルを学習するための設定です。

ChemBERTa-2の特徴:
- SMILES専用のトークナイゼーション
- RoBERTaアーキテクチャ（BERTの改良版）
- 化合物特性予測への転移学習が容易
- 大規模化合物データでの事前学習

既存のBERTベースと比較した改善点:
1. アーキテクチャ: RoBERTa（BERTより高性能）
2. トークナイゼーション: SMILES専用語彙（612トークン）
3. 学習率: 6e-5 (化合物データに最適化)
4. バッチサイズ: 128 (大きめ)
5. 最大長: 256 (SMILES文字列に最適)
"""

import os


from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer
from molcrawl.config.paths import COMPOUNDS_DATASET_DIR

# Model configuration
model_size = "small"  # Choose between small, medium, large
model_path = os.path.join(
    os.environ.get("LEARNING_SOURCE_DIR", "learning_source_20251210"),
    "compounds",
    "chemberta2-output",
    f"chemberta2-{model_size}",
)

# ChemBERTa-2 optimized settings
max_length = 256  # SMILES sequences (typical length ~50-150)
dataset_dir = COMPOUNDS_DATASET_DIR  # Organix13 dataset
learning_rate = 6e-5  # ChemBERTa-2推奨値 (RoBERTaベース)
weight_decay = 0.01  # 正則化
max_steps = 300000  # 学習ステップ数（Organix13データセット: ~10M samples）
warmup_steps = 10000  # ウォームアップステップ

log_interval = 100
save_steps = 5000  # チェックポイント保存間隔

# Batch size settings
# 化合物データは比較的小さいため、大きめのバッチサイズ
batch_size = 128
per_device_eval_batch_size = 128
gradient_accumulation_steps = 1  # Effective batch size = 128

# Tokenizer setup
# SMILES専用トークナイザーを使用
# -----------------------------------------------------------------------------
vocab_file = "assets/molecules/vocab.txt"
print(f"📖 Loading SMILES tokenizer from: {vocab_file}")

if not os.path.exists(vocab_file):
    raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")

tokenizer = CompoundsTokenizer(vocab_file, max_length)
vocab_size = len(tokenizer)
print(f"📊 Vocabulary size: {vocab_size}")
print(f"✅ Tokenizer initialized with {vocab_size} tokens")

# Calculate meta vocab size (padded to multiple of 8)
meta_vocab_size = (vocab_size // 8 + 1) * 8
print(f"📊 Meta vocab size (padded): {meta_vocab_size}")


# Preprocessing function
def preprocess_function(examples):
    """
    Add attention_mask to the dataset

    ChemBERTa-2はattention_maskを明示的に必要とします。
    既存のinput_idsからattention_maskを生成します。
    """
    if "input_ids" in examples:
        attention_masks = []
        for input_ids in examples["input_ids"]:
            # Get pad token ID
            pad_token_id = (
                tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
            )

            # Create attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1 if token_id != pad_token_id else 0 for token_id in input_ids]
            attention_masks.append(attention_mask)

        examples["attention_mask"] = attention_masks

    return examples


# Configuration summary
print("\n" + "=" * 60)
print("🧪 ChemBERTa-2 Configuration Summary")
print("=" * 60)
print(f"Model size:              {model_size}")
print(f"Model output path:       {model_path}")
print(f"Dataset directory:       {dataset_dir}")
print(f"Max sequence length:     {max_length}")
print(f"Vocabulary size:         {vocab_size}")
print(f"Meta vocab size:         {meta_vocab_size}")
print(f"Learning rate:           {learning_rate}")
print(f"Weight decay:            {weight_decay}")
print(f"Max steps:               {max_steps}")
print(f"Warmup steps:            {warmup_steps}")
print(f"Batch size:              {batch_size}")
print(f"Gradient accum steps:    {gradient_accumulation_steps}")
print(f"Effective batch size:    {batch_size * gradient_accumulation_steps}")
print(f"Save steps:              {save_steps}")
print("=" * 60 + "\n")
