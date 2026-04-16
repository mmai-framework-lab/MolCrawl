# BERT (large) fine-tuning config for ClinVar genome sequences
#
# Continues from the genome_sequence BERT pretraining checkpoint using the
# ClinVar fine-tuning dataset (reference + variant sequences from human
# clinical variant annotations).
#
# Based on bert/configs/genome_sequence.py — key differences:
#   - dataset_dir / model_path point to the ClinVar dataset and output
#   - pretrain_model_path loads weights from genome_sequence pretraining
#   - max_steps reduced to 60000 (fine-tuning, not pretraining from scratch)
#   - learning_rate reduced to 1e-5

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from molcrawl.config.paths import (
    CLINVAR_DATASET_DIR,
    get_bert_output_path,
    get_custom_tokenizer_path,
    get_refseq_tokenizer_path,
)

model_size = "large"
model_path = get_bert_output_path("genome_sequence_clinvar", model_size)
# Pretraining checkpoint to initialise weights from when no fine-tune checkpoint exists.
pretrain_model_path = get_bert_output_path("genome_sequence", model_size)

max_length = 1024
dataset_dir = CLINVAR_DATASET_DIR

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

# Tokenizer instantiation (same SentencePiece BPE as genome_sequence pretraining)
# -----------------------------------------------------------------------------
sp = spm.SentencePieceProcessor(model_file=get_refseq_tokenizer_path())
vocab_size = sp.get_piece_size()
vocab = [sp.id_to_piece(i) for i in range(vocab_size)]

tmp_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tmp_tokenizer.add_tokens(vocab)

tmp_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tmp_tokenizer)
tmp_tokenizer.unk_token = "[UNK]"
tmp_tokenizer.sep_token = "[SEP]"
tmp_tokenizer.pad_token = "[PAD]"
tmp_tokenizer.cls_token = "[CLS]"
tmp_tokenizer.mask_token = "[MASK]"

_custom_tokenizer_path = get_custom_tokenizer_path("genome_sequence", "bert")
tmp_tokenizer.save_pretrained(_custom_tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(_custom_tokenizer_path)

meta_vocab_size = vocab_size


def preprocess_function(examples):
    """Add attention_mask to the dataset."""
    if "input_ids" in examples:
        pad_token_id = (
            tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
        )
        examples["attention_mask"] = [
            [1 if token_id != pad_token_id else 0 for token_id in seq] for seq in examples["input_ids"]
        ]
    return examples
