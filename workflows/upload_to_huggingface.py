#!/usr/bin/env python3
"""
Hugging Face Hub へモデルをアップロードするPythonスクリプト

このスクリプトは huggingface_hub ライブラリを使用して、
学習済みモデルを Hugging Face Hub にアップロードします。

使用方法:
    python upload_to_huggingface.py <model_path> <repo_id> [options]

詳細は --help を参照してください。
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
    print("ERROR: huggingface_hub がインストールされていません")
    print("インストール: pip install huggingface_hub")
    sys.exit(1)


# プロジェクトルートを取得
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="Hugging Face Hub へモデルをアップロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な使用
  python upload_to_huggingface.py ../gpt2-output/rna-small matsubara-riken/rna-small-gpt2

  # プライベートリポジトリとして、モデルカード付きでアップロード
  python upload_to_huggingface.py ../gpt2-output/rna-small matsubara-riken/rna-small-gpt2 \\
      --private --create-model-card --model-type gpt2
        """,
    )

    parser.add_argument("model_path", type=str, help="アップロードするモデルのパス")
    parser.add_argument("repo_id", type=str, help="Hugging Face Hub のリポジトリID (例: username/model-name)")

    parser.add_argument("--private", action="store_true", help="プライベートリポジトリとして作成")
    parser.add_argument(
        "--commit-message", type=str, default="Upload model", help="コミットメッセージ（デフォルト: 'Upload model'）"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gpt2", "bert", "dnabert2", "esm2", "rnaformer", "chemberta2"],
        help="モデルタイプ",
    )
    parser.add_argument("--tokenizer-path", type=str, help="トークナイザーのパス")
    parser.add_argument("--config-path", type=str, help="設定ファイルのパス")
    parser.add_argument("--create-model-card", action="store_true", help="モデルカードを自動生成")
    parser.add_argument("--dry-run", action="store_true", help="実際にはアップロードせず、何が行われるか表示")

    return parser.parse_args()


def detect_model_type(model_path: Path) -> Optional[str]:
    """モデルタイプを自動検出"""
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

    # config.json からモデルタイプを検出
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
    """学習データタイプを自動検出"""
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
    """チェックポイントファイルを検索"""
    files = []

    if model_path.is_file():
        files.append(model_path)
    elif model_path.is_dir():
        # PyTorch チェックポイント
        files.extend(model_path.glob("*.pt"))
        files.extend(model_path.glob("*.pth"))
        files.extend(model_path.glob("*.bin"))

        # HuggingFace 形式
        files.extend(model_path.glob("pytorch_model.bin"))
        files.extend(model_path.glob("model.safetensors"))

        # 設定ファイル
        files.extend(model_path.glob("config.json"))
        files.extend(model_path.glob("tokenizer*.json"))
        files.extend(model_path.glob("vocab*.json"))
        files.extend(model_path.glob("vocab*.txt"))
        files.extend(model_path.glob("special_tokens_map.json"))

    return sorted(set(files))


def find_tokenizer_files(model_path: Path, tokenizer_path: Optional[Path] = None) -> list[Path]:
    """トークナイザーファイルを検索"""
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
    """モデルカード（README.md）を生成"""
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    date_str = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    cite_key = model_name.replace("-", "_")

    # タグを生成
    tags = ["pytorch"]
    if model_type:
        tags.append(model_type)
    if data_type:
        data_tag = data_type.lower().replace("/", "-").replace(" ", "-")
        tags.append(data_tag)

    # ライセンス（必要に応じて変更）
    license_str = "apache-2.0"

    # パイプラインタグを決定
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
    """モデルを Hugging Face Hub にアップロード"""

    api = HfApi()

    # モデルタイプを自動検出
    if not model_type:
        model_type = detect_model_type(model_path)
        if model_type:
            print(f"[INFO] モデルタイプを自動検出: {model_type}")

    # データタイプを自動検出
    data_type = detect_data_type(model_path)
    if data_type:
        print(f"[INFO] データタイプを自動検出: {data_type}")

    # HuggingFace互換チェックポイントディレクトリを探す
    upload_path = model_path
    latest_checkpoint_dir = find_latest_checkpoint_dir(model_path)
    if latest_checkpoint_dir:
        print(f"[INFO] HuggingFace互換チェックポイントを検出: {latest_checkpoint_dir.name}")
        upload_path = latest_checkpoint_dir

    # アップロードするファイルを検索
    checkpoint_files = find_checkpoint_files(upload_path)
    tokenizer_files = find_tokenizer_files(upload_path, tokenizer_path)

    all_files = list(set(checkpoint_files + tokenizer_files))

    if not all_files:
        print(f"[ERROR] アップロードするファイルが見つかりません: {model_path}")
        return False

    print(f"\n[INFO] アップロード対象ファイル ({len(all_files)} 件):")
    for f in all_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    if dry_run:
        print("\n[DRY-RUN] 以下の操作が実行されます:")
        print(f"  1. リポジトリ作成: {repo_id} (private={private})")
        print(f"  2. ファイルアップロード: {len(all_files)} 件")
        if create_model_card:
            print("  3. モデルカード (README.md) 生成")
        print("\n[DRY-RUN] 実際のアップロードは行いませんでした")
        return True

    # リポジトリを作成（存在しない場合）
    print(f"\n[INFO] リポジトリを確認/作成中: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"[SUCCESS] リポジトリ準備完了: {repo_id}")
    except Exception as e:
        print(f"[ERROR] リポジトリの作成に失敗: {e}")
        return False

    # ディレクトリ全体をアップロード
    if upload_path.is_dir():
        print(f"\n[INFO] ディレクトリをアップロード中: {upload_path}")
        try:
            # training_state.binはアップロードしない（学習再開用のファイルで、HF互換ではない）
            upload_folder(
                folder_path=str(upload_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["*.tfevents.*", "*.csv", "__pycache__", "*.pyc", "training_state.bin"],
            )
            print("[SUCCESS] ディレクトリのアップロード完了")
        except Exception as e:
            print(f"[ERROR] アップロードに失敗: {e}")
            return False
    else:
        # 単一ファイルをアップロード
        print(f"\n[INFO] ファイルをアップロード中: {upload_path.name}")
        try:
            upload_file(
                path_or_fileobj=str(upload_path),
                path_in_repo=upload_path.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            print("[SUCCESS] ファイルのアップロード完了")
        except Exception as e:
            print(f"[ERROR] アップロードに失敗: {e}")
            return False

    # モデルカードを生成してアップロード
    if create_model_card:
        print("\n[INFO] モデルカードを生成中...")
        model_card_content = generate_model_card(repo_id, model_type, data_type, model_path)

        try:
            api.upload_file(
                path_or_fileobj=model_card_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model card",
            )
            print("[SUCCESS] モデルカードのアップロード完了")
        except Exception as e:
            print(f"[WARNING] モデルカードのアップロードに失敗: {e}")

    print("\n[SUCCESS] アップロード完了！")
    print(f"[INFO] URL: https://huggingface.co/{repo_id}")

    return True


def main():
    """メイン関数"""
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        print(f"[ERROR] モデルパスが見つかりません: {model_path}")
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
