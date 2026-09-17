"""save_pretrained_atomic: files land complete and nothing temporary is left."""

import os

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from molcrawl.core.tokenizer_io import save_pretrained_atomic


def _tokenizer():
    vocab = {"<pad>": 0, "<mask>": 1, "[UNK]": 2, "a": 3, "b": 4}
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]")))
    tok.pad_token = "<pad>"
    tok.mask_token = "<mask>"
    tok.unk_token = "[UNK]"
    return tok


def test_saves_a_loadable_tokenizer_into_a_new_directory(tmp_path):
    target = tmp_path / "custom_tokenizer_bert"
    save_pretrained_atomic(_tokenizer(), str(target))
    loaded = AutoTokenizer.from_pretrained(str(target))
    assert loaded.mask_token_id == 1 and loaded.pad_token_id == 0


def test_overwrites_in_place_and_leaves_no_temporary_directory(tmp_path):
    target = tmp_path / "custom_tokenizer_bert"
    save_pretrained_atomic(_tokenizer(), str(target))
    save_pretrained_atomic(_tokenizer(), str(target))
    assert sorted(os.listdir(tmp_path)) == ["custom_tokenizer_bert"]
    assert "tokenizer.json" in os.listdir(target)
