"""describe_dataset / describe_tokenizer / preparation_record, and PreparedDataset.source."""

import os

import pytest

from molcrawl.models import _provenance as p


@pytest.fixture
def arrow_dir(tmp_path):
    from datasets import Dataset

    rows = [[i % 7 for i in range(12)] for _ in range(20)]
    for split in ("train", "valid"):
        Dataset.from_dict({"input_ids": rows}).save_to_disk(str(tmp_path / f"{split}.arrow"))
    return tmp_path


def test_prepared_dataset_records_how_it_opened_the_split(arrow_dir):
    from molcrawl.core.dataset import PreparedDataset

    ds = PreparedDataset(str(arrow_dir), split="train")
    assert ds.source["method"] == "load_from_disk(<dir>/<split>.arrow)"
    assert ds.source["path"] == str((arrow_dir / "train.arrow").resolve())


def test_prepared_dataset_dict_layout(tmp_path):
    from datasets import Dataset, DatasetDict

    from molcrawl.core.dataset import PreparedDataset

    rows = {"input_ids": [[1, 2, 3]] * 4}
    DatasetDict({"train": Dataset.from_dict(rows), "valid": Dataset.from_dict(rows)}).save_to_disk(str(tmp_path / "dd"))
    ds = PreparedDataset(str(tmp_path / "dd"), split="valid")
    assert ds.source["method"] == "load_from_disk(<dir>)[split]"
    assert len(ds) == 4


def test_describe_a_prepared_dataset(arrow_dir):
    from molcrawl.core.dataset import PreparedDataset

    out = p.describe_dataset(PreparedDataset(str(arrow_dir), split="train"), "train")
    assert out["rows"] == 20 and out["rows_from"] == "len(PreparedDataset)"
    assert out["loading_method"].endswith(".arrow)")
    assert out["row_length"] == 12
    assert isinstance(out["fingerprint"], str) and out["fingerprint"]
    assert out["files_opened"]["count"] >= 1


def test_describe_a_memmap_dataset(tmp_path):
    import numpy as np

    arr = np.memmap(str(tmp_path / "train.bin"), dtype=np.uint16, mode="w+", shape=(5, 8))

    class RNABinDataset:
        def __init__(self):
            self.bin_dir, self._arr, self.block = str(tmp_path), arr, 8

        def __len__(self):
            return 5

        def __getitem__(self, i):
            return self._arr[i]

    obj = RNABinDataset()
    out = p.describe_dataset(obj, "train")
    assert out["loading_method"].startswith("numpy memmap")
    assert out["path"] == os.path.abspath(str(tmp_path / "train.bin"))
    assert out["rows"] == 5 and out["row_length"] == 8


def test_describe_none_and_unknown_objects():
    assert p.describe_dataset(None, "train")["reason"] == "no dataset object"
    out = p.describe_dataset([[1, 2, 3]], "train")
    assert out["loading_method"] is None and out["source_reason"]
    assert out["fingerprint"] is None and out["fingerprint_reason"]
    assert out["row_length"] == 3


def test_describe_tokenizer_none_says_why():
    out = p.describe_tokenizer(None, vocab_size=32, special_ids={"eos_token_id": 0})
    assert out["class"] is None and out["reason"]
    assert out["vocab_size_used_by_model"] == 32
    assert out["special_token_ids_from_config"] == {"eos_token_id": 0}


def test_describe_tokenizer_hashes_its_files(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\nb\n")

    class Tok:
        name_or_path = "toy"
        vocab_file = str(vocab)
        special_tokens_map = {"pad_token": "[PAD]"}
        all_special_ids = [0]

        def __len__(self):
            return 2

    out = p.describe_tokenizer(Tok(), vocab_size=2, ambiguity_ids=[5])
    assert out["class"] == "Tok" and out["length"] == 2
    assert out["files"] == [{"attribute": "vocab_file", "path": str(vocab), "sha256": p.sha256_file(str(vocab))}]
    assert out["special_tokens_map"] == {"pad_token": "[PAD]"} and out["all_special_ids"] == [0]
    assert out["ambiguity_token_ids"] == [5]


def test_preparation_record_is_null_with_a_reason():
    out = p.preparation_record("some/dir")
    assert out["commit"] is None and out["created"] is None and out["reason"]
