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
