"""
Unit tests for tokenizer utilities.
"""

import pytest


@pytest.mark.unit
def test_tokenizer_import():
    """Test that tokenizer modules can be imported."""
    try:
        from transformers import AutoTokenizer

        assert AutoTokenizer is not None
    except ImportError:
        pytest.skip("transformers not installed")


@pytest.mark.unit
def test_basic_tokenization():
    """Test basic tokenization functionality."""
    try:
        from transformers import AutoTokenizer

        # Use a small pretrained tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        text = "Hello world"
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    except Exception as e:
        pytest.skip(f"Tokenization test skipped: {e}")


@pytest.mark.unit
def test_custom_tokenizer_loading():
    """Test loading custom tokenizer configurations."""
    # Placeholder for custom tokenizer tests
    # TODO: Implement when custom tokenizers are ready
    pass


@pytest.mark.unit
def test_molecule_nat_lang_sep_token_is_the_packing_delimiter():
    """mol_nl's sep_token must be the token prepare_gpt2 wrote between documents.

    document_masking locates per-document boundaries through
    ``tokenizer.sep_token_id``. mol_nl packs with EOS, so pointing sep_token
    anywhere else makes masking a silent no-op over the whole 1024-token block.
    mask_token must stay distinct, or every masked position would look like a
    document boundary.
    """
    try:
        from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer
    except ImportError:
        pytest.skip("molecule_nat_lang tokenizer unavailable")
    try:
        tok = MoleculeNatLangTokenizer().tokenizer
    except Exception as e:  # noqa: BLE001 — the GPT-2 tokenizer may not be staged
        pytest.skip(f"GPT-2 tokenizer not available: {e}")

    assert tok.sep_token_id is not None, "sep_token_id is required by document_masking"
    assert tok.sep_token_id == tok.eos_token_id, (
        f"sep_token_id {tok.sep_token_id} must be the packing delimiter "
        f"(EOS {tok.eos_token_id})"
    )
    assert tok.mask_token_id != tok.sep_token_id, (
        "mask_token must differ from the separator, or masking fabricates boundaries"
    )
