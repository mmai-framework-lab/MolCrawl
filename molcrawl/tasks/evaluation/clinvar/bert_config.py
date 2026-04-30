"""BERT ClinVar evaluation config (consumed via ``exec``-style use, not imported).

Exposes constants and a ``build_tokenizer()`` factory. The tokenizer build
used to run at module top level, which forced every importer (pdoc,
pytest, IDE) to have ``LEARNING_SOURCE_DIR`` set and to write a
custom-tokenizer directory under it as a side effect of import. Both
side effects are now deferred into ``build_tokenizer()`` so the module is
import-safe; the runtime entry points that actually consume the
tokenizer call ``build_tokenizer()`` after ``check_learning_source_dir``.
"""

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from molcrawl.core.paths import REFSEQ_DATASET_DIR, get_custom_tokenizer_path
from molcrawl.core.utils.environment_check import check_learning_source_dir

# BERT settings for ClinVar evaluation
model_path = "runs_train_bert_genome_sequence"
max_length = 512  # Short sequences are sufficient for ClinVar evaluation
dataset_dir = REFSEQ_DATASET_DIR
learning_rate = 6e-6
weight_decay = 1e-1
max_steps = 60000

log_interval = 100

batch_size = 16  # Use a large batch size for evaluation purposes
per_device_eval_batch_size = 8

gradient_accumulation_steps = 5 * 8

# Model size for ClinVar evaluation - use medium for good balance
model_size = "medium"


def build_tokenizer():
    """Construct the BERT tokenizer used by the ClinVar evaluation pipeline.

    Reads ``$LEARNING_SOURCE_DIR/genome_sequence/spm_tokenizer.model``,
    saves a HuggingFace-compatible custom tokenizer under
    ``get_custom_tokenizer_path("genome_sequence", "bert")``, and reloads
    it with ``AutoTokenizer.from_pretrained``. Call from script
    entry points (``run_bert_clinvar_evaluation.sh``); not safe to call
    at module import time.
    """
    learning_source_dir = check_learning_source_dir()
    tokenizer_path = f"{learning_source_dir}/genome_sequence/spm_tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

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
    return AutoTokenizer.from_pretrained(_custom_tokenizer_path), sp.get_piece_size()


# ClinVar evaluation specific settings.
# meta_vocab_size is filled in by callers after build_tokenizer() returns the
# SentencePiece piece count; left as None at import to keep this module
# import-safe.
meta_vocab_size = None

# Path setting for ClinVar evaluation
clinvar_dataset_path = "clinvar_data/clinvar_evaluation_dataset.csv"
clinvar_output_dir = "bert_clinvar_evaluation_results"
