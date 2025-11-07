import sys
import os
import json

# Add src to path
current_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
src_path = os.path.join(current_dir, "..", "..", "src")
sys.path.append(src_path)

try:
    from config.paths import get_gpt2_output_path

    model_path = get_gpt2_output_path("rna", "bert-yigarashi-2025-10-08")
    output_dir = get_gpt2_output_path("rna", "bert-yigarashi-2025-10-08")
except ImportError:
    # Fallback if config.paths is not available
    model_path = "bert-output/rna-yigarashi-2025-10-08"
    output_dir = "bert-output/rna-yigarashi-2025-10-08"

# Create output directory if it doesn't exist
os.makedirs(model_path, exist_ok=True)

# Enable custom RNA dataset loading
use_custom_rna_dataset = True

# Dataset configuration
learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR", "learning_source_20250818")
dataset_dir = f"/wren/yigarashi/molcrawl/parquet_sample_1pct"

# Try to load vocabulary file
vocab_files = [
    os.path.join(os.path.dirname(dataset_dir), "gene_vocab.json"),
    os.path.join(dataset_dir, "vocab.json"),
    os.path.join(learning_source_dir, "rna", "gene_vocab.json"),
]

rna_vocab_file = None
for vocab_file in vocab_files:
    if os.path.exists(vocab_file):
        rna_vocab_file = vocab_file
        print(f"📖 Found vocabulary file: {vocab_file}")
        break

if rna_vocab_file:
    with open(rna_vocab_file, "r") as f:
        vocab_data = json.load(f)
    meta_vocab_size = len(vocab_data)
    print(f"📊 Vocabulary size: {meta_vocab_size}")
else:
    print("⚠️ No vocabulary file found, using default size")
    meta_vocab_size = 60666  # Default size based on previous RNA experiments

# Import necessary components for custom data collator
from transformers import DataCollatorForLanguageModeling
import torch


# Create a simple tokenizer placeholder (BERT will use its own tokenizer)
class SimpleTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        # BERT tokenizer attributes
        self.mask_token = "[MASK]"
        self.mask_token_id = 3
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.cls_token = "[CLS]"
        self.cls_token_id = 1
        self.sep_token = "[SEP]"
        self.sep_token_id = 2
        self.unk_token = "[UNK]"
        self.unk_token_id = 4

    def __len__(self):
        return self.vocab_size

    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to IDs (placeholder implementation)"""
        if isinstance(tokens, str):
            return 5  # Default token ID for unknown tokens
        return [5] * len(tokens)  # Return list of default IDs

    def convert_ids_to_tokens(self, ids):
        """Convert IDs to tokens (placeholder implementation)"""
        if isinstance(ids, int):
            return f"token_{ids}"
        return [f"token_{id}" for id in ids]

    def pad(self, encoded_inputs, **kwargs):
        """Dummy pad method - data is already padded in preprocessing"""
        return encoded_inputs


tokenizer = SimpleTokenizer(meta_vocab_size)


# Custom data collator for RNA data that's already preprocessed
class RNADataCollator:
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(self, features):
        batch = {}

        # Extract input_ids and attention_mask
        batch["input_ids"] = torch.tensor(
            [f["input_ids"] for f in features], dtype=torch.long
        )
        batch["attention_mask"] = torch.tensor(
            [f["attention_mask"] for f in features], dtype=torch.long
        )

        if self.mlm:
            # Apply masking for MLM training
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"]
            )

        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        labels = inputs.clone()

        # Sample a few tokens in each sequence for MLM training
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                [
                    1
                    if token
                    in [
                        self.tokenizer.cls_token_id,
                        self.tokenizer.sep_token_id,
                        self.tokenizer.pad_token_id,
                    ]
                    else 0
                    for token in val.tolist()
                ]
                for val in labels
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# Use custom data collator
data_collator = RNADataCollator(tokenizer, mlm=True, mlm_probability=0.15)

# Training parameters optimized for RNA transcriptome data
max_length = 512  # Reasonable context length for BERT
batch_size = 4  # RNA data can be large, so smaller batch size
per_device_eval_batch_size = 1
gradient_accumulation_steps = 16  # Compensate for smaller batch size

# Training schedule
max_steps = 100000
warmup_steps = 1000  # Warmup steps
learning_rate = 3e-5  # Learning rate for BERT RNA data
weight_decay = 1e-2
log_interval = 100

# Model architecture (small model for testing)
model_size = "small"

# Dataset parameters
dataset_params = {}

# Device settings
device = "cuda" if os.path.exists("/usr/bin/nvidia-smi") else "cpu"

print(f"📋 BERT RNA Configuration Summary:")
print(f"   Dataset: {dataset_dir}")
print(f"   Vocabulary: {meta_vocab_size} tokens")
print(f"   Model: {model_size}")
print(f"   Output: {model_path}")
print(f"   Max length: {max_length}")
print(f"   Batch size: {batch_size}")
print(f"   Max steps: {max_steps}")
