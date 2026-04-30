"""
Integration tests for Compounds - Validation of actual models and data pipeline
"""

import os

import pytest


@pytest.mark.integration
@pytest.mark.compound
class TestCompoundsEndToEnd:
    """End-to-end integration tests for Compounds."""

    def test_smiles_to_scaffold_pipeline(self):
        """Test the complete SMILES → Scaffold pipeline."""
        from molcrawl.data.compounds.utils.preprocessing import prepare_scaffolds

        # Actual compound examples
        test_cases = [
            ("CCO", False),  # Ethanol - no ring
            ("c1ccccc1", True),  # Benzene - has ring
            ("CC(=O)O", False),  # Acetic acid - no ring
            ("INVALID_SMILES", False),  # Invalid
            ("", False),  # Empty - invalid
        ]

        results = []
        for smiles, should_be_valid in test_cases:
            scaffold = prepare_scaffolds(smiles)
            is_valid = scaffold != ""

            results.append(
                {
                    "smiles": smiles,
                    "scaffold": scaffold,
                    "expected_valid": should_be_valid,
                    "actual_valid": is_valid,
                    "passed": is_valid == should_be_valid,
                }
            )

        # Display results
        for result in results:
            print(
                f"SMILES: {result['smiles'][:20]:20s} | "
                f"Expected: {result['expected_valid']:5} | "
                f"Actual: {result['actual_valid']:5} | "
                f"{'✓' if result['passed'] else '✗'}"
            )

        # Verify that all test cases passed
        assert all(r["passed"] for r in results), "Some test cases failed"

    def test_batch_smiles_processing(self):
        """Verify that a large number of SMILES can be batch processed."""
        from molcrawl.data.compounds.utils.preprocessing import get_invalid_smiles_stats, prepare_scaffolds

        # Simulate a large volume of SMILES data
        test_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CC(C)C",
            "C1=CC=C(C=C1)O",
        ] * 20  # 100 SMILES

        scaffolds = []
        for smiles in test_smiles:
            scaffold = prepare_scaffolds(smiles)
            scaffolds.append(scaffold)

        # Check statistics
        invalid_count, total_count, invalid_rate, examples = get_invalid_smiles_stats()

        print("\nBatch Processing Results:")
        print(f"  Processed: {len(test_smiles)} SMILES")
        print(f"  Valid scaffolds: {len([s for s in scaffolds if s != ''])}")
        print(f"  Invalid rate: {invalid_rate:.2f}%")

        # All SMILES were processed
        assert len(scaffolds) == len(test_smiles)
        # Only SMILES with ring structures have scaffolds
        valid_count = len([s for s in scaffolds if s != ""])
        assert valid_count >= len(test_smiles) * 0.2  # ~1/5 have ring structures


@pytest.mark.integration
@pytest.mark.compound
@pytest.mark.slow
class TestCompoundsBERTIntegration:
    """Integration tests for the Compounds BERT model."""

    @pytest.fixture
    def bert_model_path(self):
        """Return the path to the BERT model (replace with the actual path)."""
        # Retrieve from environment variable, or specify the actual path
        path = os.environ.get("COMPOUNDS_BERT_MODEL_PATH")
        if path and os.path.exists(path):
            return path
        pytest.skip("BERT model path not found. Set COMPOUNDS_BERT_MODEL_PATH environment variable.")

    def test_bert_model_loading(self, bert_model_path):
        """Verify that the BERT model can be loaded correctly."""
        from transformers import BertForMaskedLM

        try:
            model = BertForMaskedLM.from_pretrained(bert_model_path)
            assert model is not None
            print(f"✓ BERT model loaded successfully from {bert_model_path}")
        except Exception as e:
            pytest.fail(f"Failed to load BERT model: {e}")

    def test_bert_tokenizer_loading(self, bert_model_path):
        """Verify that the BERT tokenizer can be loaded correctly."""
        from molcrawl.data.compounds.utils.tokenizer import SmilesTokenizer

        vocab_path = os.path.join(bert_model_path, "vocab.txt")
        if not os.path.exists(vocab_path):
            pytest.skip(f"Vocab file not found at {vocab_path}")

        try:
            tokenizer = SmilesTokenizer(vocab_path)
            assert tokenizer is not None
            print("✓ Tokenizer loaded successfully")
            print(f"  Vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            pytest.fail(f"Failed to load tokenizer: {e}")

    def test_bert_inference_pipeline(self, bert_model_path):
        """Verify that inference can be executed with the BERT model."""
        import torch
        from transformers import BertForMaskedLM

        from molcrawl.data.compounds.utils.tokenizer import SmilesTokenizer

        vocab_path = os.path.join(bert_model_path, "vocab.txt")
        if not os.path.exists(vocab_path):
            pytest.skip(f"Vocab file not found at {vocab_path}")

        try:
            # Load model and tokenizer
            model = BertForMaskedLM.from_pretrained(bert_model_path)
            tokenizer = SmilesTokenizer(vocab_path)

            model.eval()

            # Sample SMILES
            test_smiles = "CCO"  # Ethanol

            # Tokenize
            inputs = tokenizer(test_smiles, return_tensors="pt")

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)

            assert outputs is not None
            assert outputs.logits is not None
            print("✓ BERT inference successful")
            print(f"  Input SMILES: {test_smiles}")
            print(f"  Output shape: {outputs.logits.shape}")

        except Exception as e:
            pytest.fail(f"BERT inference failed: {e}")


@pytest.mark.integration
@pytest.mark.compound
@pytest.mark.slow
class TestCompoundsGPT2Integration:
    """Integration tests for the Compounds GPT2 model."""

    @pytest.fixture
    def gpt2_model_path(self):
        """Return the path to the GPT2 model (replace with the actual path)."""
        path = os.environ.get("COMPOUNDS_GPT2_MODEL_PATH")
        if path and os.path.exists(path):
            return path
        pytest.skip("GPT2 model path not found. Set COMPOUNDS_GPT2_MODEL_PATH environment variable.")

    def test_gpt2_model_loading(self, gpt2_model_path):
        """Verify that the GPT2 model can be loaded correctly."""
        from transformers import GPT2LMHeadModel

        try:
            model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
            assert model is not None
            print(f"✓ GPT2 model loaded successfully from {gpt2_model_path}")
        except Exception as e:
            pytest.fail(f"Failed to load GPT2 model: {e}")

    def test_gpt2_smiles_generation(self, gpt2_model_path):
        """Verify that SMILES can be generated with GPT2."""
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        try:
            model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)

            model.eval()

            # Start token
            prompt = "C"

            # Prepare input
            inputs = tokenizer(prompt, return_tensors="pt")

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=3,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode
            generated_smiles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            print("✓ GPT2 generation successful")
            print(f"  Prompt: {prompt}")
            print(f"  Generated {len(generated_smiles)} SMILES:")
            for i, smiles in enumerate(generated_smiles, 1):
                print(f"    {i}. {smiles[:50]}")

            assert len(generated_smiles) == 3

        except Exception as e:
            pytest.fail(f"GPT2 generation failed: {e}")

    def test_gpt2_generated_smiles_validity(self, gpt2_model_path):
        """Verify the validity of generated SMILES."""
        import torch
        from rdkit import Chem
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        try:
            model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
            model.eval()

            # Generate from multiple starting points
            prompts = ["C", "c1", "CC"]

            all_generated = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=50,
                        num_return_sequences=5,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                all_generated.extend(generated)

            # Check validity
            valid_count = 0
            for smiles in all_generated:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1

            validity_rate = (valid_count / len(all_generated)) * 100

            print("\n✓ SMILES Validity Check:")
            print(f"  Total generated: {len(all_generated)}")
            print(f"  Valid SMILES: {valid_count}")
            print(f"  Validity rate: {validity_rate:.1f}%")

            # Verify that the validity rate is above a threshold (expecting >= 50%)
            assert validity_rate >= 50, f"Validity rate too low: {validity_rate:.1f}%"

        except Exception as e:
            pytest.fail(f"Validity check failed: {e}")


@pytest.mark.integration
@pytest.mark.compound
class TestCompoundsDatasetIntegration:
    """Integration tests for the Compounds dataset."""

    def test_dataset_loading(self, mock_compounds_dataset):
        """Verify that the mock dataset can be loaded correctly."""
        import pandas as pd

        df = pd.read_csv(mock_compounds_dataset)

        assert "smiles" in df.columns
        assert len(df) > 0
        print(f"✓ Dataset loaded: {len(df)} compounds")

    def test_dataset_preprocessing(self, mock_compounds_dataset):
        """Test the dataset preprocessing pipeline."""
        import pandas as pd

        from molcrawl.data.compounds.utils.preprocessing import prepare_scaffolds

        df = pd.read_csv(mock_compounds_dataset)

        # Generate scaffolds
        df["scaffold"] = df["smiles"].apply(prepare_scaffolds)

        # Statistics
        valid_scaffolds = df[df["scaffold"] != ""]
        invalid_scaffolds = df[df["scaffold"] == ""]

        print("\n✓ Dataset Preprocessing:")
        print(f"  Total: {len(df)}")
        print(f"  Valid: {len(valid_scaffolds)}")
        print(f"  Invalid: {len(invalid_scaffolds)}")

        # Scaffolds are generated only for data containing ring structures
        assert len(valid_scaffolds) >= 1
