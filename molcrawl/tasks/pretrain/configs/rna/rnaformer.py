"""
RNAformer Configuration for RNA Transcriptome Data

This config uses an existing RNA transcriptome dataset (CellXGene).
Settings for learning RNAformer models.

Features of RNAformer:
- Tokenization specialized for gene expression data
- Cell type specific learning
- Geneformer architecture based
- Efficient long text context processing

Improvements compared to the existing BERT base:
1. Tokenization: Gene ID-based vocabulary
2. Learning rate: 1e-4 (optimized for RNA transcriptome)
3. Batch size: 8 (emphasis on memory efficiency)
4. Maximum length: 1024 (supports long gene expression profiles)
"""

import os
import json


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Use RNA dataset from refined source
RNA_REFINED_DIR = os.environ.get("LEARNING_SOURCE_DIR", "learning_source_20250904-rna-refined")
RNA_DATASET_DIR = os.path.join(RNA_REFINED_DIR, "rna")
GENE_VOCAB_PATH = os.path.join(RNA_DATASET_DIR, "gene_vocab.json")

# Model configuration
model_size = "small"  # Choose between small, medium, large
model_path = os.path.join(RNA_REFINED_DIR, "rna", "rnaformer-output", f"rnaformer-{model_size}")

# RNAformer optimized settings
max_length = 1024  # RNA transcriptome sequences
dataset_dir = os.path.join(RNA_DATASET_DIR, "training_ready_hf_dataset")
learning_rate = 1e-4  # RNAformer recommended value (higher than Geneformer)
weight_decay = 0.1  # regularization
max_steps = 100000  # Number of learning steps (adjusted according to data amount)
warmup_steps = 10000  # warmup steps

log_interval = 100
save_steps = 1000  # Checkpoint save interval

# Number of worker processes for the preprocessing .map() pass over the
# training_ready_hf_dataset (40M rows). Single-process preprocessing
# takes ~3.5 hours; 18 workers brings it under 5 minutes on this host.
preprocess_num_proc: int = 18

# Batch size settings
# Smaller batch size as RNA transcriptome uses large memory
batch_size = 8
per_device_eval_batch_size = 4
gradient_accumulation_steps = 16  # Effective batch size = 8 * 16 = 128

# Tokenizer setup
# Use RNA gene vocabulary
# -----------------------------------------------------------------------------
print(f"📖 Loading RNA gene vocabulary from: {GENE_VOCAB_PATH}")
if not os.path.exists(GENE_VOCAB_PATH):
    raise FileNotFoundError(f"Gene vocabulary not found at {GENE_VOCAB_PATH}")

with open(GENE_VOCAB_PATH, "r") as f:
    gene_vocab = json.load(f)

vocab_size = len(gene_vocab)
print(f"📊 Vocabulary size: {vocab_size}")

# Create HuggingFace compatible tokenizer
pre_tokenizer = Tokenizer(WordLevel(vocab=gene_vocab, unk_token="<unk>"))

tmp_tokenizer = PreTrainedTokenizerFast(tokenizer_object=pre_tokenizer)
tmp_tokenizer.unk_token = "<unk>"
tmp_tokenizer.sep_token = "<eos>"
tmp_tokenizer.pad_token = "<pad>"
tmp_tokenizer.cls_token = "<eos>"  # Use <eos> as CLS token for RNA
tmp_tokenizer.mask_token = "<mask>"

# Save and reload
custom_tokenizer_path = os.path.join(RNA_REFINED_DIR, "rna", "custom_tokenizer_rnaformer")
os.makedirs(custom_tokenizer_path, exist_ok=True)
tmp_tokenizer.save_pretrained(custom_tokenizer_path)

tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer_path)
print(f"✅ Tokenizer initialized with {len(tokenizer)} tokens")

# Calculate meta vocab size (padded to multiple of 8)
meta_vocab_size = (len(tokenizer) // 8 + 1) * 8
print(f"📊 Meta vocab size (padded): {meta_vocab_size}")


# Preprocessing function
def preprocess_function(examples):
    """
        Add attention_mask to the dataset

    RNAformer requires attention_mask explicitly.
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
print("🧬 RNAformer Configuration Summary")
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
