import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.config.paths import REFSEQ_DATASET_DIR

# 共通環境チェックモジュールを追加
from src.utils.environment_check import check_learning_source_dir

# ClinVar評価用BERT設定
model_path = "runs_train_bert_genome_sequence"
max_length = 512  # ClinVar評価では短いシーケンスで十分
dataset_dir = REFSEQ_DATASET_DIR
learning_rate = 6e-6
weight_decay = 1e-1
max_steps = 600000

log_interval = 100

batch_size = 16  # 評価用なので大きなバッチサイズを使用
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 8

# Tokenizer instantiation for ClinVar evaluation
# -----------------------------------------------------------------------------
# 実際に存在するトークナイザーファイルを使用
learning_source_dir = check_learning_source_dir()
tokenizer_path = f"{learning_source_dir}/genome_sequence/spm_tokenizer.model"
sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
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

# Model size for ClinVar evaluation - use medium for good balance
model_size = "medium"

# ClinVar評価固有の設定
meta_vocab_size = vocab_size

# ClinVar評価用のパス設定
clinvar_dataset_path = "clinvar_data/clinvar_evaluation_dataset.csv"
clinvar_output_dir = "bert_clinvar_evaluation_results"
