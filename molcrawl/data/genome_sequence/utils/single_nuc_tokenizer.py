"""Single-nucleotide HuggingFace tokenizer for the subset/Evo2 flow.

The Phase 3 parquet builder
(:mod:`molcrawl.data.genome_sequence.dataset.refseq.raw_to_parquet_single_nuc`)
stores *pre-tokenised* ``input_ids`` using a fixed 10-symbol vocabulary:

    A=0 T=1 G=2 C=3 N=4 [PAD]=5 [UNK]=6 [CLS]=7 [SEP]=8 [MASK]=9

This module builds (and caches on disk) a matching
``PreTrainedTokenizerFast`` so that:

* ``DataCollatorForLanguageModeling`` has the right ``mask_token_id`` /
  ``pad_token_id`` / ``all_special_ids`` for dynamic MLM masking at train time
  (no mask is baked into the parquet — see Phase 3 design doc).
* Anyone who wants to tokenise raw DNA text downstream gets the same id
  assignment as the parquet.

The tokenizer is character-level: every input character becomes one token via
``WordLevel`` lookup, with unknown bytes mapping to ``[UNK]``.
"""

from pathlib import Path
from typing import Union

# Vocabulary (must match raw_to_parquet_single_nuc constants exactly).
NUC_TO_ID = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

VOCAB = {
    "A": 0, "T": 1, "G": 2, "C": 3, "N": 4,
    PAD_TOKEN: 5, UNK_TOKEN: 6, CLS_TOKEN: 7, SEP_TOKEN: 8, MASK_TOKEN: 9,
}
VOCAB_SIZE = 10


def build_single_nuc_tokenizer(save_dir: Union[str, Path]):
    """Build (or load) the single-nucleotide HF tokenizer and return it.

    ``save_dir`` receives ``tokenizer.json`` + the standard ``PreTrainedTokenizerFast``
    artefacts so that subsequent training/inference can use
    ``AutoTokenizer.from_pretrained(save_dir)``.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    save_dir = Path(save_dir)
    if (save_dir / "tokenizer.json").exists():
        return AutoTokenizer.from_pretrained(str(save_dir))

    inner = Tokenizer(WordLevel(vocab=dict(VOCAB), unk_token=UNK_TOKEN))
    # Character-level: split every input string between each character so each
    # char becomes one WordLevel lookup. ``Split("", isolated)`` would error
    # on empty patterns in some versions; the regex ``""`` with behavior
    # "isolated" is the documented way to split on every character.
    inner.pre_tokenizer = Split(pattern="", behavior="isolated")

    hf = PreTrainedTokenizerFast(
        tokenizer_object=inner,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        cls_token=CLS_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(str(save_dir))
    # Re-load through AutoTokenizer so callers get the same object type they'd
    # see when loading from a checkpoint.
    return AutoTokenizer.from_pretrained(str(save_dir))
