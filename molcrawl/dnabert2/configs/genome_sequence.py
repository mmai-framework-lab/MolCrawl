"""
DNABERT-2 Configuration for Genome Sequence Data

This config uses the existing genome_sequence dataset (REFSEQ)
Settings for learning the DNABERT-2 model.

Features of DNABERT-2:
- BPE (Byte Pair Encoding) Tokenization
- More efficient training
- Optimization specific to DNA sequences

Improvements compared to the existing BERT base:
1. Tokenization: No k-mer required, efficient with BPE
2. Learning rate: 3e-5 (higher than BERT)
3. Batch size: 16 (larger)
4. Maximum length: 512 (BERT is 1024, but DNABERT is efficient at 512)
"""

import os


import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from molcrawl.core.paths import REFSEQ_DATASET_DIR, get_refseq_tokenizer_path

# Model configuration
model_size = "small"  # Choose between small, medium, large
model_path = os.path.join(
    os.environ.get("LEARNING_SOURCE_DIR", "learning_source_20251210"),
    "genome_sequence",
    "dnabert2-output",
    f"dnabert2-{model_size}",
)

# DNABERT-2 optimized settings
max_length = 1024  # datasetaccording to1024Set to (REFSEQ dataset uses 1024）
dataset_dir = REFSEQ_DATASET_DIR
learning_rate = 3e-5  # DNABERT-2 recommended value (higher than BERT's 6e-6)
weight_decay = 0.01  # regularization
max_steps = 200000  # Number of learning steps (adjusted according to data amount)

log_interval = 100
save_steps = 5000  # Checkpoint save interval

# Batch size settings
# DNABERT-2 is efficient with larger batch sizes
batch_size = 16
per_device_eval_batch_size = 8
gradient_accumulation_steps = 4  # Effective batch size = 16 * 4 = 64

# Tokenizer setup
# Reuse existing SentencePiece tokenizer
# DNABERT-2 is BPE-based, but existing SPM tokenizers can also be used
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
custom_tokenizer_path = os.path.join(
    os.environ.get("LEARNING_SOURCE_DIR", "learning_source_20251210"),
    "genome_sequence",
    "custom_tokenizer_dnabert2",
)
os.makedirs(custom_tokenizer_path, exist_ok=True)
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

    DNABERT-2 requires attention_mask explicitly.
    Generate attention_mask from existing input_ids.
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
print("🧬 DNABERT-2 Configuration Summary")
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
print(f"Batch size:              {batch_size}")
print(f"Gradient accum steps:    {gradient_accumulation_steps}")
print(f"Effective batch size:    {batch_size * gradient_accumulation_steps}")
print(f"Save steps:              {save_steps}")
print("=" * 60 + "\n")
