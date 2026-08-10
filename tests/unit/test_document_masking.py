"""Tests for intra-block (document) masking of packed sequences."""

import torch

from molcrawl.models._collators.document_masking import (
    DocumentMaskingCollator,
    document_attention_mask,
    document_ids,
    document_position_ids,
)

SEP = 13


def test_separator_belongs_to_the_document_it_ends():
    # two molecules: [a b SEP] [c d SEP]
    ids = torch.tensor([[20, 21, SEP, 22, 23, SEP]])
    assert document_ids(ids, SEP).tolist() == [[0, 0, 0, 1, 1, 1]]


def test_block_without_separators_is_one_document():
    ids = torch.tensor([[20, 21, 22, 23]])
    assert document_ids(ids, SEP).tolist() == [[0, 0, 0, 0]]


def test_attention_mask_is_block_diagonal():
    ids = torch.tensor([[20, SEP, 21, SEP]])
    mask = document_attention_mask(ids, SEP)
    assert mask.shape == (1, 4, 4)
    assert mask[0].tolist() == [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]


def test_no_position_can_see_another_document():
    ids = torch.tensor([[20, 21, SEP, 22, 23, 24, SEP]])
    doc = document_ids(ids, SEP)
    mask = document_attention_mask(ids, SEP)
    cross = (doc.unsqueeze(2) != doc.unsqueeze(1)) & mask.bool()
    assert not bool(cross.any())


def test_every_position_can_see_its_own_document():
    ids = torch.tensor([[20, 21, SEP, 22, 23, SEP]])
    doc = document_ids(ids, SEP)
    mask = document_attention_mask(ids, SEP)
    same = (doc.unsqueeze(2) == doc.unsqueeze(1)) & ~mask.bool()
    assert not bool(same.any())


def test_position_ids_restart_at_each_document():
    ids = torch.tensor([[20, 21, SEP, 22, 23, 24, SEP]])
    assert document_position_ids(ids, SEP).tolist() == [[0, 1, 2, 0, 1, 2, 3]]


def test_batch_rows_are_independent():
    ids = torch.tensor([[20, SEP, 21, 22], [20, 21, 22, SEP]])
    doc = document_ids(ids, SEP)
    assert doc.tolist() == [[0, 0, 1, 1], [0, 0, 0, 0]]


def test_collator_rewrites_attention_and_adds_positions():
    def base(features):
        ids = torch.tensor([f["input_ids"] for f in features])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.full_like(ids, -100),
        }

    coll = DocumentMaskingCollator(base, separator_id=SEP)
    batch = coll([{"input_ids": [20, SEP, 21, SEP]}])

    assert batch["attention_mask"].shape == (1, 4, 4)  # 1D mask replaced by 3D
    assert batch["position_ids"].tolist() == [[0, 1, 0, 1]]
    assert batch["labels"].shape == (1, 4)  # base collator's work is untouched


def test_collator_can_leave_position_ids_alone():
    def base(features):
        ids = torch.tensor([f["input_ids"] for f in features])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    coll = DocumentMaskingCollator(base, separator_id=SEP, reset_position_ids=False)
    batch = coll([{"input_ids": [20, SEP, 21, SEP]}])

    assert "position_ids" not in batch
    assert batch["attention_mask"].shape == (1, 4, 4)
