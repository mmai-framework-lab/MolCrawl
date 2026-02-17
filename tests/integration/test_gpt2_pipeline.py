"""
Integration tests for GPT2 model pipeline.
"""

import pytest


@pytest.mark.integration
@pytest.mark.gpt2
def test_gpt2_model_initialization():
    """Test GPT2 model can be initialized."""
    try:
        from transformers import GPT2Model, GPT2Config

        config = GPT2Config(
            vocab_size=1000,
            n_positions=512,
            n_embd=128,
            n_layer=2,
            n_head=2,
        )

        model = GPT2Model(config)
        assert model is not None

    except ImportError:
        pytest.skip("transformers not installed")


@pytest.mark.integration
@pytest.mark.gpt2
@pytest.mark.slow
def test_gpt2_generation():
    """Test GPT2 text generation."""
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Config

        config = GPT2Config(
            vocab_size=1000,
            n_positions=512,
            n_embd=128,
            n_layer=2,
            n_head=2,
        )

        model = GPT2LMHeadModel(config)
        model.eval()

        # Create dummy input
        input_ids = torch.randint(0, 1000, (1, 5))

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=10,
                do_sample=False,
            )

        assert outputs is not None
        assert outputs.shape[1] == 10

    except ImportError as e:
        pytest.skip(f"Required packages not installed: {e}")
