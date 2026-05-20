"""Unit tests for molcrawl.models._collators.ambiguity_aware_collator."""

from __future__ import annotations

import pytest
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from molcrawl.models._collators.ambiguity_aware_collator import (
    GENOME_AMBIGUOUS_TOKENS,
    IGNORE_INDEX,
    PROTEIN_AMBIGUOUS_TOKENS,
    AmbiguityAwareMLMCollator,
    ambiguous_tokens_for_modality,
    make_mlm_collator,
    mask_ambiguous_targets_for_clm,
    resolve_ambiguous_token_ids,
)


# -----------------------------------------------------------------------------
# Tokenizer fixtures
# -----------------------------------------------------------------------------
def _build_char_tokenizer(chars):
    """Build a simple character-level HF tokenizer over the given chars plus
    standard special tokens."""
    vocab = {tok: i for i, tok in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(chars)
    )}
    inner = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    inner.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture
def genome_tokenizer():
    # Standard bases + N + IUPAC ambiguity codes
    return _build_char_tokenizer(["A", "C", "G", "T"] + GENOME_AMBIGUOUS_TOKENS)


@pytest.fixture
def protein_tokenizer():
    aas = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
           "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
           "U", "O"]  # incl. Sec & Pyl (real AAs)
    return _build_char_tokenizer(aas + PROTEIN_AMBIGUOUS_TOKENS)


# -----------------------------------------------------------------------------
# resolve_ambiguous_token_ids
# -----------------------------------------------------------------------------
def test_resolve_ambiguous_token_ids_all_present(genome_tokenizer):
    ids = resolve_ambiguous_token_ids(
        genome_tokenizer, GENOME_AMBIGUOUS_TOKENS, log=False
    )
    assert len(ids) == len(GENOME_AMBIGUOUS_TOKENS)
    # All ids should be distinct and not the unk id
    assert len(set(ids)) == len(ids)
    assert genome_tokenizer.unk_token_id not in ids


def test_resolve_ambiguous_token_ids_filters_unk(genome_tokenizer):
    """Unknown tokens get filtered (resolve to unk and are skipped)."""
    ids = resolve_ambiguous_token_ids(
        genome_tokenizer, ["N", "ZZZ_NOT_IN_VOCAB"], log=False
    )
    assert len(ids) == 1  # only "N" survives


# -----------------------------------------------------------------------------
# MLM wrapper — collator output structure
# -----------------------------------------------------------------------------
def _make_examples(tokenizer, sequences):
    return [tokenizer(s, return_special_tokens_mask=True) for s in sequences]


def test_mlm_collator_returns_input_ids_and_labels(genome_tokenizer):
    collator = AmbiguityAwareMLMCollator(
        genome_tokenizer,
        ambiguous_tokens=GENOME_AMBIGUOUS_TOKENS,
        mlm_probability=0.5,
    )
    examples = _make_examples(genome_tokenizer, ["A C G T N A C G T"])
    batch = collator(examples)
    assert "input_ids" in batch
    assert "labels" in batch


def test_mlm_collator_zeros_loss_on_ambiguous_positions(genome_tokenizer):
    """The label at any position whose original input was an ambiguous token
    (N here) must be IGNORE_INDEX so it contributes zero loss."""
    # Force high masking probability so we exercise both the masked and
    # non-masked branches.
    torch.manual_seed(0)
    collator = AmbiguityAwareMLMCollator(
        genome_tokenizer,
        ambiguous_tokens=GENOME_AMBIGUOUS_TOKENS,
        mlm_probability=0.99,
    )

    # 32 sequences of pure N → guaranteed to hit ambiguous positions.
    seqs = ["N N N N N N N N N N N N"] * 32
    examples = _make_examples(genome_tokenizer, seqs)
    batch = collator(examples)

    n_id = genome_tokenizer.convert_tokens_to_ids("N")
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Every position where input is still N OR labels still N → must be IGNORE.
    # (After 80/10/10 masking, input may be [MASK], random, or N; labels
    # holds the original N for masked, IGNORE for non-masked.)
    # Easiest check: no position should have label == n_id.
    assert (labels == n_id).sum().item() == 0


def test_mlm_collator_does_not_block_normal_tokens(genome_tokenizer):
    """Loss should still be computed for non-ambiguous tokens at masked
    positions — i.e. labels should contain SOME non-IGNORE entries."""
    torch.manual_seed(0)
    collator = AmbiguityAwareMLMCollator(
        genome_tokenizer,
        ambiguous_tokens=GENOME_AMBIGUOUS_TOKENS,
        mlm_probability=0.5,
    )
    seqs = ["A C G T A C G T A C G T A C G T"] * 16
    examples = _make_examples(genome_tokenizer, seqs)
    batch = collator(examples)
    labels = batch["labels"]
    # At least some positions should retain a real label (not IGNORE).
    assert (labels != IGNORE_INDEX).sum().item() > 0


def test_protein_collator_does_not_mask_u_and_o(protein_tokenizer):
    """U (Sec) and O (Pyl) are real amino acids — they must NOT be excluded."""
    torch.manual_seed(0)
    collator = AmbiguityAwareMLMCollator(
        protein_tokenizer,
        ambiguous_tokens=PROTEIN_AMBIGUOUS_TOKENS,
        mlm_probability=0.99,
    )
    seqs = ["U O U O U O U O U O U O"] * 32
    examples = _make_examples(protein_tokenizer, seqs)
    batch = collator(examples)

    u_id = protein_tokenizer.convert_tokens_to_ids("U")
    o_id = protein_tokenizer.convert_tokens_to_ids("O")
    labels = batch["labels"]
    # U and O should appear as labels — they should be predicted normally.
    assert (labels == u_id).sum().item() > 0 or (labels == o_id).sum().item() > 0


def test_protein_collator_excludes_x_b_z(protein_tokenizer):
    torch.manual_seed(0)
    collator = AmbiguityAwareMLMCollator(
        protein_tokenizer,
        ambiguous_tokens=PROTEIN_AMBIGUOUS_TOKENS,
        mlm_probability=0.99,
    )
    seqs = ["X B Z X B Z X B Z X B Z"] * 32
    examples = _make_examples(protein_tokenizer, seqs)
    batch = collator(examples)
    x_id = protein_tokenizer.convert_tokens_to_ids("X")
    b_id = protein_tokenizer.convert_tokens_to_ids("B")
    z_id = protein_tokenizer.convert_tokens_to_ids("Z")
    labels = batch["labels"]
    assert (labels == x_id).sum().item() == 0
    assert (labels == b_id).sum().item() == 0
    assert (labels == z_id).sum().item() == 0


# -----------------------------------------------------------------------------
# CLM helper
# -----------------------------------------------------------------------------
def test_clm_helper_sets_ignore_index_for_ambiguous_positions():
    targets = torch.tensor([[10, 11, 99, 12, 13, 99, 14]])
    ambig_ids = [99]
    out = mask_ambiguous_targets_for_clm(targets, ambig_ids)
    expected = torch.tensor([[10, 11, IGNORE_INDEX, 12, 13, IGNORE_INDEX, 14]])
    assert torch.equal(out, expected)


def test_clm_helper_returns_unchanged_when_no_ids():
    targets = torch.tensor([[1, 2, 3, 4]])
    out = mask_ambiguous_targets_for_clm(targets, [])
    assert torch.equal(out, targets)


def test_clm_helper_does_not_mutate_input():
    targets = torch.tensor([[10, 99, 12]])
    out = mask_ambiguous_targets_for_clm(targets, [99])
    # Original unchanged
    assert targets[0, 1].item() == 99
    # Output rewrote position
    assert out[0, 1].item() == IGNORE_INDEX


# -----------------------------------------------------------------------------
# Factory + modality dispatch
# -----------------------------------------------------------------------------
def test_make_mlm_collator_returns_base_when_empty(genome_tokenizer):
    coll = make_mlm_collator(genome_tokenizer, ambiguous_tokens=[])
    # Returns base DataCollatorForLanguageModeling, not the wrapper
    from transformers import DataCollatorForLanguageModeling
    assert isinstance(coll, DataCollatorForLanguageModeling)
    assert not isinstance(coll, AmbiguityAwareMLMCollator)


def test_make_mlm_collator_returns_wrapper_when_nonempty(genome_tokenizer):
    coll = make_mlm_collator(genome_tokenizer, ambiguous_tokens=["N"])
    assert isinstance(coll, AmbiguityAwareMLMCollator)


def test_ambiguous_tokens_for_modality():
    assert ambiguous_tokens_for_modality("genome_sequence") == GENOME_AMBIGUOUS_TOKENS
    assert ambiguous_tokens_for_modality("protein_sequence") == PROTEIN_AMBIGUOUS_TOKENS
    assert ambiguous_tokens_for_modality("compounds") == []
    assert ambiguous_tokens_for_modality("rna") == []
    assert ambiguous_tokens_for_modality("unknown_modality") == []
