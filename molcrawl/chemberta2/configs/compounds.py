"""
ChemBERTa-2 Configuration for SMILES Compounds Data

This config uses an existing compounds dataset (Organix13)
Settings for learning the ChemBERTa-2 model.

Features of ChemBERTa-2:
- Tokenization exclusively for SMILES
- RoBERTa architecture (improved version of BERT)
- Easy transfer learning to compound property prediction
- Pre-training on large-scale compound data

Improvements compared to the existing BERT base:
1. Architecture: RoBERTa (higher performance than BERT)
2. Tokenization: SMILES exclusive vocabulary (612 tokens)
3. Learning rate: 6e-5 (optimized for compound data)
4. Batch size: 128 (large)
5. Maximum length: 256 (ideal for SMILES strings)
"""

import os


from molcrawl.compounds.utils.tokenizer import CompoundsTokenizer
from molcrawl.core.paths import COMPOUNDS_DATASET_DIR

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
learning_rate = 6e-5  # ChemBERTa-2 recommended value (based on RoBERTa)
weight_decay = 0.01  # regularization
max_steps = 300000  # Number of learning steps (Organix13 dataset: ~10M samples)
warmup_steps = 10000  # warmup steps

log_interval = 100
save_steps = 5000  # Checkpoint save interval

# Batch size settings
# Larger batch size as compound data is relatively small
batch_size = 128
per_device_eval_batch_size = 128
gradient_accumulation_steps = 1  # Effective batch size = 128

# Tokenizer setup
# Use SMILES dedicated tokenizer
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

    ChemBERTa-2 requires attention_mask explicitly.
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
