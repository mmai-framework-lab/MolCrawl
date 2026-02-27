import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from molcrawl.config.paths import REFSEQ_DATASET_DIR, get_refseq_tokenizer_path, get_bert_output_path

model_size = "small"  # Choose between small, medium or large
model_path = get_bert_output_path("genome_sequence", model_size)
max_length = 1024
dataset_dir = REFSEQ_DATASET_DIR
learning_rate = 6e-6
weight_decay = 1e-1
max_steps = 600000

log_interval = 100
save_steps = 100  # Save checkpoint every 100 steps instead of default 1000

batch_size = 8
per_device_eval_batch_size = 1

gradient_accumulation_steps = 5 * 16

# Tokenizer instantiation
# -----------------------------------------------------------------------------
sp = spm.SentencePieceProcessor(model_file=get_refseq_tokenizer_path())
# Get vocabulary size
vocab_size = sp.get_piece_size()

# Get all tokens in the vocabulary
vocab = [sp.id_to_piece(i) for i in range(vocab_size)]


tmp_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tmp_tokenizer.add_tokens(vocab)

tmp_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tmp_tokenizer)
tmp_tokenizer.unk_token = "[UNK]"
tmp_tokenizer.sep_token = "[SEP]"
tmp_tokenizer.pad_token = "[PAD]"
tmp_tokenizer.cls_token = "[CLS]"
tmp_tokenizer.mask_token = "[MASK]"

tmp_tokenizer.save_pretrained("custom_tokenizer")

tokenizer = AutoTokenizer.from_pretrained("custom_tokenizer")


# Add preprocessing function to create attention_mask
def preprocess_function(examples):
    """Add attention_mask to the dataset"""
    if "input_ids" in examples:
        # Create attention_mask: 1 for real tokens, 0 for padding
        attention_masks = []
        for input_ids in examples["input_ids"]:
            # Assuming pad_token_id is tokenizer.pad_token_id or 0
            pad_token_id = (
                tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None else 0
            )
            attention_mask = [1 if token_id != pad_token_id else 0 for token_id in input_ids]
            attention_masks.append(attention_mask)

        examples["attention_mask"] = attention_masks

    return examples
