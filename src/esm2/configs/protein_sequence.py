"""
ESM-2 Configuration for Protein Sequence Data

このconfigは既存のprotein_sequenceデータセット（UniProt）を使用して
ESM-2モデルを学習するための設定です。

ESM-2の特徴:
- タンパク質配列専用の進化的スケールモデリング
- 6.5億パラメータまでスケール可能
- Structure prediction, function annotation等に高性能

既存のBERTベースと比較した改善点:
1. アーキテクチャ: ESM専用に最適化
2. 学習率: 4e-4 (BERTより高め)
3. Dropout: 0.0 (ESM-2の論文設定)
4. Position embeddings: Learned
5. 収束: より高速で安定
"""

import os


from typing import Any, Dict, List

import torch
from transformers import DataCollatorForLanguageModeling

from config.paths import UNIPROT_DATASET_DIR
from protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer

# Model configuration
model_size = "small"  # Choose between small, medium, large
model_path = os.path.join(
    os.environ.get("LEARNING_SOURCE_DIR", "learning_source_20251210"),
    "protein_sequence",
    "esm2-output",
    f"esm2-{model_size}"
)

# ESM-2 optimized settings
max_length = 1024  # ESM-2標準値（タンパク質配列の長さに対応）
dataset_dir = UNIPROT_DATASET_DIR
learning_rate = 4e-4  # ESM-2推奨値（論文より）
weight_decay = 0.01
max_steps = 500000  # 学習ステップ数（データ量に応じて調整）
warmup_steps = 2000  # ウォームアップステップ

log_interval = 100
save_steps = 5000  # チェックポイント保存間隔

# Batch size settings
# タンパク質配列は長いため、バッチサイズは小さめに設定
batch_size = 4
per_device_eval_batch_size = 2
gradient_accumulation_steps = 32  # Effective batch size = 4 * 32 = 128

# Tokenizer setup
# 既存のESMトークナイザーを使用（BERT互換ラッパー）
# -----------------------------------------------------------------------------
print("📖 Creating ESM protein tokenizer...")
tokenizer = create_bert_protein_tokenizer()
print(f"✅ Tokenizer initialized with {len(tokenizer.get_vocab())} tokens")

# Get vocabulary size
meta_vocab_size = len(tokenizer.get_vocab())
print(f"📊 Vocabulary size: {meta_vocab_size}")


# Preprocessing function to add attention_mask
def preprocess_function(examples):
    """
    Add attention_mask to dataset for ESM-2 compatibility
    """
    # Handle batch processing
    if "input_ids" in examples:
        input_ids = examples["input_ids"]

        # Create attention_mask (1 for real tokens, 0 for padding)
        if isinstance(input_ids[0], list):  # Batch of sequences
            attention_masks = []
            for seq in input_ids:
                # Get pad token ID
                pad_token_id = (
                    tokenizer.pad_token_id
                    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None
                    else 0
                )
                attention_mask = [1 if token != pad_token_id else 0 for token in seq]
                attention_masks.append(attention_mask)
            examples["attention_mask"] = attention_masks
        else:  # Single sequence
            pad_token_id = (
                tokenizer.pad_token_id
                if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None
                else 0
            )
            examples["attention_mask"] = [1 if token != pad_token_id else 0 for token in input_ids]

    # Handle sequence_tokens field (if exists, rename to input_ids)
    elif "sequence_tokens" in examples:
        examples["input_ids"] = examples["sequence_tokens"]
        # Recursively call to add attention_mask
        return preprocess_function(examples)

    return examples


# Custom data collator for protein sequences
class ProteinSequenceDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for protein sequences that handles field name conversion
    """

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Override to handle any remaining field name issues
        """
        # Convert any sequence_tokens to input_ids before processing
        for example in examples:
            if "sequence_tokens" in example and "input_ids" not in example:
                example["input_ids"] = example.pop("sequence_tokens")

        # Call parent method
        return super().torch_call(examples)


# Use custom data collator with MLM probability 0.15 (ESM-2 standard)
data_collator = ProteinSequenceDataCollator(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


# Configuration summary
print("\n" + "="*60)
print("🧬 ESM-2 Configuration Summary")
print("="*60)
print(f"Model size:              {model_size}")
print(f"Model output path:       {model_path}")
print(f"Dataset directory:       {dataset_dir}")
print(f"Max sequence length:     {max_length}")
print(f"Vocabulary size:         {meta_vocab_size}")
print(f"Learning rate:           {learning_rate}")
print(f"Weight decay:            {weight_decay}")
print(f"Max steps:               {max_steps}")
print(f"Warmup steps:            {warmup_steps}")
print(f"Batch size:              {batch_size}")
print(f"Gradient accum steps:    {gradient_accumulation_steps}")
print(f"Effective batch size:    {batch_size * gradient_accumulation_steps}")
print(f"Save steps:              {save_steps}")
print("="*60 + "\n")
