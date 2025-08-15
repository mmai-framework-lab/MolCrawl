from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import sentencepiece as spm

model_path = "runs_train_bert_genome_sequence"
max_length = 1024
dataset_dir = "learning_source_202508/refseq/training_ready_hf_dataset"
learning_rate = 6e-6
weight_decay = 1e-1
max_steps = 600000

log_interval = 100

batch_size = 8
per_device_eval_batch_size = 1

gradient_accumulation_steps = 5 * 16

# Tokenizer instantiation
# -----------------------------------------------------------------------------
sp = spm.SentencePieceProcessor(model_file="learning_source_202508/refseq/spm_tokenizer.model")
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

# Choose between small, medium or large
model_size = "small"
output_dir = "out-bert-genome-sequence"
