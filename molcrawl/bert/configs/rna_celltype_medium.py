# BERT (medium) fine-tuning config for RNA cell type annotation
#
# Continues from the rna BERT pretraining checkpoint using the
# Geneformer cell type annotation dataset (human single-cell transcriptomes
# pre-tokenized as rank-value gene encodings from ctheodoris/Genecorpus-30M).
#
# Based on bert/configs/rna.py — key differences:
#   - dataset_dir / model_path point to the celltype dataset and output
#   - pretrain_model_path loads weights from rna pretraining
#   - max_steps reduced to 60000 (fine-tuning, not pretraining from scratch)
#   - learning_rate reduced to 1e-5

from typing import Dict, List

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from molcrawl.core.paths import (
    RNA_CELLTYPE_DATASET_DIR,
    get_bert_output_path,
    get_custom_tokenizer_path,
)
from molcrawl.rna.dataset.geneformer.tokenizer import TranscriptomeTokenizer

# Reconstruct the same WordLevel tokenizer used during RNA pretraining
original_tokenizer = TranscriptomeTokenizer()

_pre_tokenizer = Tokenizer(WordLevel(vocab=original_tokenizer.gene_token_dict, unk_token="[UNK]"))
_tmp_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=_pre_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
)
_tmp_tokenizer.unk_token = "[UNK]"
_tmp_tokenizer.sep_token = "[SEP]"
_tmp_tokenizer.pad_token = "<pad>"
_tmp_tokenizer.cls_token = "[CLS]"
_tmp_tokenizer.mask_token = "<mask>"

_custom_tokenizer_path = get_custom_tokenizer_path("rna", "bert")
_tmp_tokenizer.save_pretrained(_custom_tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(_custom_tokenizer_path)

# Round up to nearest multiple of 8 to match pretraining model
meta_vocab_size = (len(original_tokenizer) // 8 + 1) * 8

model_size = "medium"
# Fine-tuning checkpoint output — separate from pretraining output
model_path = get_bert_output_path("rna_celltype", model_size)
# Pretraining checkpoint to initialise weights from when no fine-tune checkpoint exists.
pretrain_model_path = get_bert_output_path("rna", model_size)

max_length = 1024
dataset_dir = RNA_CELLTYPE_DATASET_DIR

# Fine-tuning hyper-parameters (lower LR and fewer steps than pretraining)
learning_rate = 1e-5
max_steps = 60000  # ~10 % of the 600k pretraining steps
weight_decay = 1e-1

log_interval = 100
save_steps = 1000
early_stopping_patience = 3  # Stop after 3 evals (300 steps) with no improvement

batch_size = 8
per_device_eval_batch_size = 8
gradient_accumulation_steps = 5 * 16


def preprocess_function(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    """Add attention_mask and token_type_ids to the dataset."""
    if "input_ids" in examples:
        pad_token_id = (
            tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
        )
        examples["attention_mask"] = [
            [1 if token_id != pad_token_id else 0 for token_id in seq] for seq in examples["input_ids"]
        ]
        # Add token_type_ids (all zeros for single segment)
        examples["token_type_ids"] = [[0] * len(seq) for seq in examples["input_ids"]]
    return examples


# Special Tokens
eos_token: int = 0  # eos
