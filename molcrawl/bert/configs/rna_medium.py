# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

from typing import Dict, List

# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from molcrawl.core.paths import CELLXGENE_DATASET_DIR, get_bert_output_path, get_custom_tokenizer_path

# Build the tokenizer using the WordLevel model
from molcrawl.rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

original_tokenizer = TranscriptomeTokenizer()

pre_tokenizer = Tokenizer(WordLevel(vocab=original_tokenizer.gene_token_dict, unk_token="[UNK]"))

# Wrap into Hugging Face tokenizer
tmp_tokenizer = PreTrainedTokenizerFast(tokenizer_object=pre_tokenizer, unk_token="[UNK]", pad_token="[PAD]")
tmp_tokenizer.unk_token = "[UNK]"
tmp_tokenizer.sep_token = "[SEP]"
tmp_tokenizer.pad_token = "<pad>"
tmp_tokenizer.cls_token = "[CLS]"
tmp_tokenizer.mask_token = "<mask>"

_custom_tokenizer_path = get_custom_tokenizer_path("rna", "bert")
tmp_tokenizer.save_pretrained(_custom_tokenizer_path)

tokenizer = AutoTokenizer.from_pretrained(_custom_tokenizer_path)


max_steps: int = 60000
early_stopping = False  # Pretraining: run the full schedule, no early stopping
# early_stopping_patience: int = 3  # N/A when early_stopping = Falsevement
model_size: str = "medium"  # Choose between small, medium or large
model_path: str = get_bert_output_path("rna", model_size)
max_length: int = 1024
dataset_dir: str = CELLXGENE_DATASET_DIR
learning_rate: float = 6e-6
weight_decay: float = 1e-1
log_interval: int = 100
save_steps: int = 100  # Save checkpoint every 100 steps instead of default 1000

batch_size: int = 8
per_device_eval_batch_size: int = 8
gradient_accumulation_steps: int = 5 * 16

# Parallel preprocessing for attention_mask creation
preprocess_num_proc: int = 18


# Add preprocessing function to create attention_mask
def preprocess_function(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    """Add attention_mask and token_type_ids to the dataset"""
    if "input_ids" in examples:
        # Create attention_mask: 1 for real tokens, 0 for padding
        attention_masks: List[List[int]] = []
        token_type_ids_list: List[List[int]] = []
        for input_ids in examples["input_ids"]:
            # Assuming pad_token_id is tokenizer.pad_token_id or 0
            pad_token_id: int = (
                tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
            )
            attention_mask = [1 if token_id != pad_token_id else 0 for token_id in input_ids]
            attention_masks.append(attention_mask)
            # Add token_type_ids (all zeros for single segment)
            token_type_ids_list.append([0] * len(input_ids))

        examples["attention_mask"] = attention_masks
        examples["token_type_ids"] = token_type_ids_list

    return examples


# Special Tokens
eos_token: int = 0  # eos
