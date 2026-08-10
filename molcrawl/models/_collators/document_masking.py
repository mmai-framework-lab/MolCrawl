"""Restrict attention to one document at a time inside a packed block.

Packing concatenates many documents into one fixed-length block:

    [mol_0] [SEP] [mol_1] [SEP] ... [mol_25] [SEP]

Nothing stops attention from crossing those boundaries, so a masked token in
``mol_3`` can attend to the other ~25 molecules sharing its block. Every modality
packs this way and none applies intra-block masking.

This collator confines each position to its own document and restarts position ids
at every boundary, which is what the block would look like if the documents had
been batched separately — without giving up packing's efficiency or changing the
sequence length, so BERT keeps the same data, batch and block size as GPT-2 and only
the objective differs.

The separator token itself belongs to the document it terminates.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def document_ids(input_ids: torch.Tensor, separator_id: int) -> torch.Tensor:
    """Number each position with the index of the document it belongs to.

    A separator terminates its document, so it carries that document's id and the
    next position starts the following one.
    """
    is_sep = input_ids == separator_id
    # Shift right: a position belongs to the next document only *after* a separator.
    started = torch.zeros_like(is_sep, dtype=torch.long)
    started[:, 1:] = is_sep[:, :-1].long()
    return started.cumsum(dim=1)


def document_attention_mask(input_ids: torch.Tensor, separator_id: int) -> torch.Tensor:
    """Block-diagonal attention mask: 1 where two positions share a document."""
    doc = document_ids(input_ids, separator_id)
    return (doc.unsqueeze(2) == doc.unsqueeze(1)).long()


def document_position_ids(input_ids: torch.Tensor, separator_id: int) -> torch.Tensor:
    """Position ids that restart at 0 for each document in the block."""
    doc = document_ids(input_ids, separator_id)
    absolute = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(doc)
    # First absolute position of each document, broadcast back over its positions.
    starts = torch.zeros_like(doc)
    boundary = torch.zeros_like(doc, dtype=torch.bool)
    boundary[:, 0] = True
    boundary[:, 1:] = doc[:, 1:] != doc[:, :-1]
    starts = torch.cummax(torch.where(boundary, absolute, torch.zeros_like(absolute)), dim=1).values
    return absolute - starts


class DocumentMaskingCollator:
    """Wrap an MLM collator so packed blocks attend within a document only.

    The base collator still decides what to mask; this only rewrites
    ``attention_mask`` into its 3D block-diagonal form and adds ``position_ids``.
    ``BertModel.get_extended_attention_mask`` accepts a 3D mask directly.
    """

    def __init__(self, base_collator, separator_id: int, reset_position_ids: bool = True):
        self.base_collator = base_collator
        self.separator_id = separator_id
        self.reset_position_ids = reset_position_ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(features)
        input_ids = batch["input_ids"]
        batch["attention_mask"] = document_attention_mask(input_ids, self.separator_id)
        if self.reset_position_ids:
            batch["position_ids"] = document_position_ids(input_ids, self.separator_id)
        return batch
