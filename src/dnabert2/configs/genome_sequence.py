"""
DNABERT-2 Configuration for Genome Sequence Data

このconfigは既存のgenome_sequenceデータセット（REFSEQ）を使用して
DNABERT-2モデルを学習するための設定です。

DNABERT-2の特徴:
- BPE (Byte Pair Encoding) トークナイゼーション
- より効率的なトレーニング
- DNA配列に特化した最適化

既存のBERTベースと比較した改善点:
1. トークナイゼーション: k-mer不要、BPEで効率的
2. 学習率: 3e-5 (BERTより高め)
3. バッチサイズ: 16 (より大きめ)
4. 最大長: 512 (BERTは1024だが、DNABERTは512で効率的)
"""

import os


import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from config.paths import REFSEQ_DATASET_DIR, get_refseq_tokenizer_path

# Model configuration
model_size = "small"  # Choose between small, medium, large
model_path = os.path.join(
    os.environ.get("LEARNING_SOURCE_DIR", "learning_source_20251210"),
    "genome_sequence",
    "dnabert2-output",
    f"dnabert2-{model_size}"
)

# DNABERT-2 optimized settings
max_length = 1024  # データセットに合わせて1024に設定（REFSEQ dataset uses 1024）
dataset_dir = REFSEQ_DATASET_DIR
learning_rate = 3e-5  # DNABERT-2推奨値（BERTの6e-6より高い）
weight_decay = 0.01  # 正則化
max_steps = 200000  # 学習ステップ数（データ量に応じて調整）

log_interval = 100
save_steps = 5000  # チェックポイント保存間隔

# Batch size settings
# DNABERT-2はより大きいバッチサイズで効率的
batch_size = 16
per_device_eval_batch_size = 8
gradient_accumulation_steps = 4  # Effective batch size = 16 * 4 = 64

# Tokenizer setup
# 既存のSentencePieceトークナイザーを再利用
# DNABERT-2はBPEベースだが、既存のSPMトークナイザーも使用可能
# -----------------------------------------------------------------------------
print(f"📖 Loading SentencePiece tokenizer from: {get_refseq_tokenizer_path()}")
sp = spm.SentencePieceProcessor(model_file=get_refseq_tokenizer_path())
vocab_size = sp.get_piece_size()
print(f"📊 Vocabulary size: {vocab_size}")

# Get all tokens in the vocabulary
vocab = [sp.id_to_piece(i) for i in range(vocab_size)]

# Create HuggingFace compatible tokenizer
tmp_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tmp_tokenizer.add_tokens(vocab)

tmp_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tmp_tokenizer)
tmp_tokenizer.unk_token = "[UNK]"
tmp_tokenizer.sep_token = "[SEP]"
tmp_tokenizer.pad_token = "[PAD]"
tmp_tokenizer.cls_token = "[CLS]"
tmp_tokenizer.mask_token = "[MASK]"

# Save and reload
custom_tokenizer_path = "custom_tokenizer_dnabert2"
tmp_tokenizer.save_pretrained(custom_tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_path)

print(f"✅ Tokenizer initialized with {len(tokenizer)} tokens")

# Calculate meta_vocab_size (padded to multiple of 8 for efficiency)
meta_vocab_size = (len(tokenizer) // 8 + 1) * 8
print(f"📊 Meta vocab size (padded): {meta_vocab_size}")


# Preprocessing function
def preprocess_function(examples):
    """
    Add attention_mask to the dataset

    DNABERT-2はattention_maskを明示的に必要とします。
    既存のinput_idsからattention_maskを生成します。
    """
    if "input_ids" in examples:
        attention_masks = []
        for input_ids in examples["input_ids"]:
            # Get pad token ID
            pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0

            # Create attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1 if token_id != pad_token_id else 0 for token_id in input_ids]
            attention_masks.append(attention_mask)

        examples["attention_mask"] = attention_masks

    return examples


# Configuration summary
print("\n" + "="*60)
print("🧬 DNABERT-2 Configuration Summary")
print("="*60)
print(f"Model size:              {model_size}")
print(f"Model output path:       {model_path}")
print(f"Dataset directory:       {dataset_dir}")
print(f"Max sequence length:     {max_length}")
print(f"Vocabulary size:         {vocab_size}")
print(f"Meta vocab size:         {meta_vocab_size}")
print(f"Learning rate:           {learning_rate}")
print(f"Weight decay:            {weight_decay}")
print(f"Max steps:               {max_steps}")
print(f"Batch size:              {batch_size}")
print(f"Gradient accum steps:    {gradient_accumulation_steps}")
print(f"Effective batch size:    {batch_size * gradient_accumulation_steps}")
print(f"Save steps:              {save_steps}")
print("="*60 + "\n")
