"""Shared data collators for MolCrawl pretraining."""

from molcrawl.models._collators.document_masking import (
    DocumentMaskingCollator,
    document_attention_mask,
    document_ids,
    document_position_ids,
)
from molcrawl.models._collators.ambiguity_aware_collator import (
    GENOME_AMBIGUOUS_TOKENS,
    PROTEIN_AMBIGUOUS_TOKENS,
    AmbiguityAwareMLMCollator,
    ambiguous_tokens_for_modality,
    infer_modality_from_path,
    make_mlm_collator,
    mask_ambiguous_targets_for_clm,
    resolve_ambiguous_token_ids,
)

__all__ = [
    "DocumentMaskingCollator",
    "document_attention_mask",
    "document_ids",
    "document_position_ids",
    "GENOME_AMBIGUOUS_TOKENS",
    "PROTEIN_AMBIGUOUS_TOKENS",
    "AmbiguityAwareMLMCollator",
    "ambiguous_tokens_for_modality",
    "infer_modality_from_path",
    "make_mlm_collator",
    "mask_ambiguous_targets_for_clm",
    "resolve_ambiguous_token_ids",
]
