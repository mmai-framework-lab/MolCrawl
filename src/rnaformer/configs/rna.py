"""
RNAformer Configuration for RNA Transcriptome Data

このconfigは既存のRNA transcriptomeデータセット（CellXGene）を使用して
RNAformerモデルを学習するための設定です。

RNAformerの特徴:
- 遺伝子発現データに特化したトークナイゼーション
- セルタイプ特異的な学習
- Geneformerアーキテクチャベース
- 効率的な長文コンテキスト処理

既存のBERTベースと比較した改善点:
1. トークナイゼーション: 遺伝子IDベースの語彙
2. 学習率: 1e-4 (RNA transcriptomeに最適化)
3. バッチサイズ: 8 (メモリ効率重視)
4. 最大長: 1024 (長い遺伝子発現プロファイル対応)
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
learning_rate = 1e-4  # RNAformer推奨値 (Geneformerより高め)
weight_decay = 0.1  # 正則化
max_steps = 100000  # 学習ステップ数（データ量に応じて調整）
warmup_steps = 10000  # ウォームアップステップ

log_interval = 100
save_steps = 1000  # チェックポイント保存間隔

# Batch size settings
# RNA transcriptomeは大きなメモリを使用するため、小さめのバッチサイズ
batch_size = 8
per_device_eval_batch_size = 4
gradient_accumulation_steps = 16  # Effective batch size = 8 * 16 = 128

# Tokenizer setup
# RNA遺伝子語彙を使用
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
custom_tokenizer_path = "custom_tokenizer_rnaformer"
if not os.path.exists(custom_tokenizer_path):
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

    RNAformerはattention_maskを明示的に必要とします。
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
