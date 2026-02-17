"""
Integration tests for BERT model pipeline.
"""

import pytest


@pytest.mark.integration
@pytest.mark.bert
def test_bert_model_initialization():
    """Test BERT model can be initialized."""
    try:
        from transformers import BertModel, BertConfig

        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
        )

        model = BertModel(config)
        assert model is not None

    except ImportError:
        pytest.skip("transformers not installed")


@pytest.mark.integration
@pytest.mark.bert
@pytest.mark.slow
def test_bert_forward_pass():
    """Test BERT model forward pass."""
    try:
        import torch
        from transformers import BertModel, BertConfig

        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
        )

        model = BertModel(config)
        model.eval()

        # Create dummy input
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.last_hidden_state is not None
        assert outputs.last_hidden_state.shape == (2, 10, 128)

    except ImportError as e:
        pytest.skip(f"Required packages not installed: {e}")
