#!/usr/bin/env python3


# 設定ファイルから定数をインポート
from src.config.paths import UNIPROT_DATASET_DIR

try:
    from datasets import load_from_disk

    dataset = load_from_disk(UNIPROT_DATASET_DIR)
    print("Dataset columns:", dataset["train"].column_names)

    sample = dataset["train"][0]
    print("Sample keys:", list(sample.keys()))

    for key in sample.keys():
        value = sample[key]
        if isinstance(value, list):
            print(f"{key}: list[{len(value)}] - {value[:5]}...")
        else:
            print(f"{key}: {type(value)} = {value}")

    # Check if attention_mask exists
    if "attention_mask" not in sample:
        print("WARNING: attention_mask not found in dataset!")
        print("This may cause issues with DataCollatorForLanguageModeling")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
