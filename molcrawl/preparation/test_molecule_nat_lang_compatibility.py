#!/usr/bin/env python3
"""
Test script to verify molecule_nat_lang data compatibility with BERT and GPT-2 training scripts
"""

import os
import sys

# Add project root to path

import torch
from datasets import load_from_disk

from molcrawl.core.dataset import PreparedDataset


def test_bert_compatibility():
    """Test if data is compatible with BERT training"""
    print("=" * 70)
    print("Testing BERT Compatibility")
    print("=" * 70)

    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR", "learning_20251121")
    dataset_dir = f"{learning_source_dir}/molecule_nat_lang/arrow_splits"

    try:
        # Load train dataset
        train_data = load_from_disk(f"{dataset_dir}/train.arrow")
        print(f"✅ Loaded train dataset: {len(train_data)} samples")

        # Check required fields
        required_fields = ["input_ids", "attention_mask"]
        has_all = all(field in train_data.column_names for field in required_fields)

        if has_all:
            print(f"✅ Has all required fields: {required_fields}")
        else:
            missing = [f for f in required_fields if f not in train_data.column_names]
            print(f"❌ Missing fields: {missing}")
            return False

        # Test sample
        sample = train_data[0]
        print(f"✅ Sample has {len(sample['input_ids'])} input tokens")
        print(f"✅ Sample has {len(sample['attention_mask'])} attention mask values")

        # Verify data types
        assert isinstance(sample["input_ids"], list), "input_ids should be list"
        assert isinstance(sample["attention_mask"], list), "attention_mask should be list"
        print("✅ Data types are correct")

        return True

    except Exception as e:
        print(f"❌ BERT compatibility test failed: {e}")
        return False


def test_gpt2_compatibility():
    """Test if data is compatible with GPT-2 training"""
    print("\n" + "=" * 70)
    print("Testing GPT-2 Compatibility")
    print("=" * 70)

    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR", "learning_20251121")
    dataset_dir = f"{learning_source_dir}/molecule_nat_lang/arrow_splits"

    try:
        # Use PreparedDataset (same as train.py)
        train_data = PreparedDataset(dataset_dir, split="train")
        print(f"✅ Loaded train dataset: {len(train_data)} samples")

        # Test __getitem__
        sample = train_data[0]
        print(f"✅ Sample type: {type(sample)}")
        print(f"✅ Sample shape: {sample.shape}")
        print(f"✅ Sample dtype: {sample.dtype}")

        # Verify it's a tensor
        assert isinstance(sample, torch.Tensor), "Sample should be torch.Tensor"
        assert sample.dtype == torch.long, "Sample should be torch.long"
        print("✅ Data format is correct for GPT-2")

        # Test batch loading with padding (as done in train.py)
        print("\n📦 Testing batch loading with padding/truncation...")
        batch_indices = [0, 1, 2]
        block_size = 512  # Same as in config

        sequences = [train_data[i] for i in batch_indices]
        padded_sequences = []
        for seq in sequences:
            if len(seq) > block_size:
                padded_sequences.append(seq[:block_size])
            elif len(seq) < block_size:
                padding = torch.zeros(block_size - len(seq), dtype=torch.long)
                padded_sequences.append(torch.cat([seq, padding]))
            else:
                padded_sequences.append(seq)

        batch = torch.stack(padded_sequences)
        print(f"✅ Batch shape after padding: {batch.shape}")
        print("✅ Batch processing works correctly (as in train.py)")

        return True

    except Exception as e:
        print(f"❌ GPT-2 compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("Molecule NL Data Compatibility Test")
    print("=" * 70 + "\n")

    bert_ok = test_bert_compatibility()
    gpt2_ok = test_gpt2_compatibility()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"BERT compatibility: {'✅ PASS' if bert_ok else '❌ FAIL'}")
    print(f"GPT-2 compatibility: {'✅ PASS' if gpt2_ok else '❌ FAIL'}")

    if bert_ok and gpt2_ok:
        print("\n✅ All tests passed! Data is compatible with both BERT and GPT-2.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
