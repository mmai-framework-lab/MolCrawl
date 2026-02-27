#!/usr/bin/env python3
"""
GuacaMol Dataset Download Script

GuacaMolベンチマークデータセットをFigshareからダウンロードします。
https://figshare.com/projects/GuacaMol/56639

使用方法:
    python src/preparation/download_guacamol.py

環境変数:
    LEARNING_SOURCE_DIR: データセット保存先のベースディレクトリ（必須）
"""

import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# GuacaMol dataset URLs from Figshare
GUACAMOL_URLS = {
    "train": "https://figshare.com/ndownloader/files/13612760",  # guacamol_v1_train.smiles
    "valid": "https://figshare.com/ndownloader/files/13612766",  # guacamol_v1_valid.smiles
    "test": "https://figshare.com/ndownloader/files/13612757",  # guacamol_v1_test.smiles
}


def download_file(url, output_path, chunk_size=8192):
    """
    URLからファイルをダウンロードして保存

    Args:
        url: ダウンロード元URL
        output_path: 保存先パス
        chunk_size: チャンクサイズ（バイト）
    """
    output_path = Path(output_path)

    # 既に存在する場合はスキップ
    if output_path.exists():
        print(f"✓ Already exists: {output_path.name}")
        return True

    print(f"Downloading {output_path.name}...")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # ファイルサイズを取得
        total_size = int(response.headers.get("content-length", 0))

        # プログレスバー付きでダウンロード
        with (
            open(output_path, "wb") as f,
            tqdm(
                desc=output_path.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ Downloaded: {output_path.name}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {output_path.name}: {e}", file=sys.stderr)
        # 失敗したファイルを削除
        if output_path.exists():
            output_path.unlink()
        return False


def download_guacamol(compounds_dir):
    """
    GuacaMolデータセットをダウンロード

    Args:
        compounds_dir: compoundsディレクトリのパス（例: learning_source_XXX/compounds）

    Raises:
        RuntimeError: ダウンロードに失敗した場合
    """
    import logging

    logger = logging.getLogger(__name__)

    output_dir = Path(compounds_dir) / "benchmark" / "GuacaMol"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading GuacaMol benchmark from https://figshare.com/projects/GuacaMol/56639")
    logger.info(f"Destination: {output_dir}")

    success_count = 0
    total_count = len(GUACAMOL_URLS)

    for split, url in GUACAMOL_URLS.items():
        filename = f"guacamol_v1_{split}.smiles"
        output_path = output_dir / filename

        if download_file(url, output_path):
            success_count += 1

    if success_count < total_count:
        raise RuntimeError(f"GuacaMol download incomplete: {success_count}/{total_count} files downloaded")

    logger.info(f"✓ GuacaMol: All {total_count} files downloaded successfully")


def main():
    """GuacaMolデータセットをダウンロード（スタンドアロン実行用）"""

    # LEARNING_SOURCE_DIRの確認
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source_dir:
        print(
            "ERROR: Environment variable 'LEARNING_SOURCE_DIR' is not set.",
            file=sys.stderr,
        )
        print(
            "Please set LEARNING_SOURCE_DIR before running this script:",
            file=sys.stderr,
        )
        print("  export LEARNING_SOURCE_DIR='learning_20251104'", file=sys.stderr)
        sys.exit(1)

    compounds_dir = Path(learning_source_dir) / "compounds"

    try:
        download_guacamol(str(compounds_dir))
        print("\nNext steps:")
        print("  1. Run the GPT-2 preparation script:")
        print(
            f"     LEARNING_SOURCE_DIR={learning_source_dir} python src/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml"
        )
        return 0
    except RuntimeError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
