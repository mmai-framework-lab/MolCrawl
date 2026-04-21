#!/usr/bin/env python3
"""
Python script to upload model to Hugging Face Hub

This script uses the huggingface_hub library to
Upload the trained model to Hugging Face Hub.

How to use:
    python upload_to_huggingface.py <model_path> <repo_id> [options]

See --help for details.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Installation: pip install huggingface_hub")
    sys.exit(1)


# get project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Upload model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  # Basic usage
  python upload_to_huggingface.py ../gpt2-output/rna-small your-username/rna-small-gpt2

  # Upload as a private repository with model card
  python upload_to_huggingface.py ../gpt2-output/rna-small your-username/rna-small-gpt2 \\
      --private --create-model-card --model-type gpt2
        """,
    )

    parser.add_argument("model_path", type=str, help="Path of model to upload")
    parser.add_argument("repo_id", type=str, help="Hugging Face Hub repository ID (e.g. username/model-name)")

    parser.add_argument("--private", action="store_true", help="Create as private repository")
    parser.add_argument("--commit-message", type=str, default="Upload model", help="Commit message (default: 'Upload model')")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gpt2", "bert", "dnabert2", "esm2", "rnaformer", "chemberta2"],
        help="Model type",
    )
    parser.add_argument("--tokenizer-path", type=str, help="Tokenizer path")
    parser.add_argument("--config-path", type=str, help="Configuration file path")
    parser.add_argument("--create-model-card", action="store_true", help="Automatically generate model card")
    parser.add_argument("--dry-run", action="store_true", help="Do not actually upload, show what will happen")

    return parser.parse_args()


def detect_model_type(model_path: Path) -> Optional[str]:
    """Auto detect model type"""
    path_str = str(model_path).lower()

    if "gpt2" in path_str or model_path.name.startswith("gpt2"):
        return "gpt2"
    elif "bert" in path_str:
        if "dnabert" in path_str:
            return "dnabert2"
        elif "chemberta" in path_str:
            return "chemberta2"
        else:
            return "bert"
    elif "esm" in path_str:
        return "esm2"
    elif "rnaformer" in path_str:
        return "rnaformer"

    # Detect model type from config.json
    config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            model_type = config.get("model_type", "")
            if model_type:
                return model_type
        except (json.JSONDecodeError, KeyError):
            pass

    return None


def detect_data_type(model_path: Path) -> Optional[str]:
    """Automatically detect training data type"""
    path_str = str(model_path).lower()

    if "rna" in path_str:
        return "RNA"
    elif "protein" in path_str:
        return "Protein"
    elif "genome" in path_str or "dna" in path_str:
        return "DNA/Genome"
    elif "compound" in path_str or "molecule" in path_str or "smiles" in path_str:
        return "Molecule/Compound"
    elif "molecule_nat_lang" in path_str:
        return "Molecule-NL"

    return None


def find_latest_checkpoint_dir(model_path: Path) -> Optional[Path]:
    """
    Find the latest HuggingFace format checkpoint directory (checkpoint-{step}/).
    Returns the path to the latest checkpoint directory, or None if not found.
    """
    if not model_path.is_dir():
        return None

    checkpoint_dirs = []
    for entry in model_path.iterdir():
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            try:
                step = int(entry.name.split("-")[1])
                # Check if it has required HF files
                config_json = entry / "config.json"
                pytorch_model = entry / "pytorch_model.bin"
                if config_json.exists() and pytorch_model.exists():
                    checkpoint_dirs.append((step, entry))
            except (ValueError, IndexError):
                continue

    if not checkpoint_dirs:
        return None

    # Sort by step number and return the latest
    checkpoint_dirs.sort(reverse=True)
    return checkpoint_dirs[0][1]


def find_checkpoint_files(model_path: Path) -> list[Path]:
    """Search checkpoint file"""
    files = []

    if model_path.is_file():
        files.append(model_path)
    elif model_path.is_dir():
        # PyTorch Checkpoint
        files.extend(model_path.glob("*.pt"))
        files.extend(model_path.glob("*.pth"))
        files.extend(model_path.glob("*.bin"))

        # HuggingFace form
        files.extend(model_path.glob("pytorch_model.bin"))
        files.extend(model_path.glob("model.safetensors"))

        # configuration file
        files.extend(model_path.glob("config.json"))
        files.extend(model_path.glob("tokenizer*.json"))
        files.extend(model_path.glob("vocab*.json"))
        files.extend(model_path.glob("vocab*.txt"))
        files.extend(model_path.glob("special_tokens_map.json"))

    return sorted(set(files))


def find_tokenizer_files(model_path: Path, tokenizer_path: Optional[Path] = None) -> list[Path]:
    """Search tokenizer file"""
    files = []

    search_paths = [model_path] if model_path.is_dir() else [model_path.parent]
    if tokenizer_path:
        search_paths.insert(0, tokenizer_path if tokenizer_path.is_dir() else tokenizer_path.parent)

    for search_path in search_paths:
        files.extend(search_path.glob("tokenizer*.json"))
        files.extend(search_path.glob("vocab*.json"))
        files.extend(search_path.glob("vocab*.txt"))
        files.extend(search_path.glob("special_tokens_map.json"))
        files.extend(search_path.glob("added_tokens.json"))
        files.extend(search_path.glob("merges.txt"))

    return sorted(set(files))


def generate_model_card(
    repo_id: str,
    model_type: Optional[str],
    data_type: Optional[str],
    model_path: Path,
) -> str:
    """Generate model card (README.md)"""
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    date_str = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    cite_key = model_name.replace("-", "_")

    # generate tag
    tags = ["pytorch"]
    if model_type:
        tags.append(model_type)
    if data_type:
        data_tag = data_type.lower().replace("/", "-").replace(" ", "-")
        tags.append(data_tag)

    # License (change as necessary)
    license_str = "apache-2.0"

    # Determine pipeline tag
    pipeline_tag = "text-generation"
    if model_type in ["bert", "dnabert2", "esm2"]:
        pipeline_tag = "fill-mask"

    tags_yaml = "\n".join([f"- {tag}" for tag in tags])

    card_content = f"""---
license: {license_str}
tags:
{tags_yaml}
pipeline_tag: {pipeline_tag}
---

# {model_name}

## Model Description

This model was trained using the RIKEN Foundation Model training pipeline.

- **Model Type**: {model_type or "Unknown"}
- **Data Type**: {data_type or "Unknown"}
- **Training Date**: {date_str}

## Usage

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Example usage
inputs = tokenizer("your input sequence", return_tensors="pt")
outputs = model(**inputs)
```

## Training

This model was trained with the RIKEN Foundation Model pipeline.
For more details, please refer to the training configuration files included in this repository.

## License

This model is released under the {license_str.upper()} license.

## Citation

If you use this model, please cite:

```bibtex
@misc{{{cite_key},
  title={{{model_name}}},
  author={{{{RIKEN}}}},
  year={{{current_year}}},
  publisher={{{{Hugging Face}}}},
  url={{{{https://huggingface.co/{repo_id}}}}}
}}
```
"""
    return card_content


def upload_model(
    model_path: Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload model",
    model_type: Optional[str] = None,
    tokenizer_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    create_model_card: bool = False,
    dry_run: bool = False,
) -> bool:
    """Upload model to Hugging Face Hub"""

    api = HfApi()

    # Automatically detect model type
    if not model_type:
        model_type = detect_model_type(model_path)
        if model_type:
            print(f"[INFO] Auto detect model type: {model_type}")

    # Auto detect data type
    data_type = detect_data_type(model_path)
    if data_type:
        print(f"[INFO] Auto detect data type: {data_type}")

    # Find HuggingFace compatible checkpoint directory
    upload_path = model_path
    latest_checkpoint_dir = find_latest_checkpoint_dir(model_path)
    if latest_checkpoint_dir:
        print(f"[INFO] HuggingFace compatible checkpoint detected: {latest_checkpoint_dir.name}")
        upload_path = latest_checkpoint_dir

    # Search for files to upload
    checkpoint_files = find_checkpoint_files(upload_path)
    tokenizer_files = find_tokenizer_files(upload_path, tokenizer_path)

    all_files = list(set(checkpoint_files + tokenizer_files))

    if not all_files:
        print(f"[ERROR] Unable to find file to upload: {model_path}")
        return False

    print(f"\n[INFO] Files to be uploaded ({len(all_files)}):")
    for f in all_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    if dry_run:
        print("\n[DRY-RUN] The following operations will be performed:")
        print(f" 1. Create repository: {repo_id} (private={private})")
        print(f" 2. File uploads: {len(all_files)} items")
        if create_model_card:
            print(" 3. Model card (README.md) generation")
        print("\n[DRY-RUN] No actual upload was performed")
        return True

    # Create a repository (if it doesn't exist)
    print(f"\n[INFO] Checking/creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"[SUCCESS] Repository ready: {repo_id}")
    except Exception as e:
        print(f"[ERROR] Failed to create repository: {e}")
        return False

    # Upload entire directory
    if upload_path.is_dir():
        print(f"\n[INFO] Uploading directory: {upload_path}")
        try:
            # Do not upload training_state.bin (this is a file for resuming learning and is not HF compatible)
            upload_folder(
                folder_path=str(upload_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["*.tfevents.*", "*.csv", "__pycache__", "*.pyc", "training_state.bin"],
            )
            print("[SUCCESS] Directory upload completed")
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")
            return False
    else:
        # Upload a single file
        print(f"\n[INFO] Uploading file: {upload_path.name}")
        try:
            upload_file(
                path_or_fileobj=str(upload_path),
                path_in_repo=upload_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            print("[SUCCESS] File upload completed")
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")
            return False

    # Generate and upload model card
    if create_model_card:
        print("\n[INFO] Generating model card...")
        model_card_content = generate_model_card(repo_id, model_type, data_type, model_path)

        try:
            api.upload_file(
                path_or_fileobj=model_card_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model card",
            )
            print("[SUCCESS] Model card upload completed")
        except Exception as e:
            print(f"[WARNING] Failed to upload model card: {e}")

    print("\n[SUCCESS] Upload completed!")
    print(f"[INFO] URL: https://huggingface.co/{repo_id}")

    return True


def main():
    """Main function"""
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        print(f"[ERROR] Model path not found: {model_path}")
        sys.exit(1)

    tokenizer_path = None
    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path)
        if not tokenizer_path.is_absolute():
            tokenizer_path = Path.cwd() / tokenizer_path
        tokenizer_path = tokenizer_path.resolve()

    config_path = None
    if args.config_path:
        config_path = Path(args.config_path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        config_path = config_path.resolve()

    success = upload_model(
        model_path=model_path,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        model_type=args.model_type,
        tokenizer_path=tokenizer_path,
        config_path=config_path,
        create_model_card=args.create_model_card,
        dry_run=args.dry_run,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
