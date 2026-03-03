#!/usr/bin/env python3
"""
画像管理ユーティリティ

データセット準備・検証の結果画像を統一的に管理するためのユーティリティ
"""

import os
from typing import TypedDict

from molcrawl.utils.environment_check import check_learning_source_dir


def get_image_output_dir(model_type: str) -> str:
    """
    指定されたモデルタイプの画像出力ディレクトリを取得

    Args:
        model_type (str): モデルタイプ ('protein_sequence', 'genome_sequence', 'compounds', 'rna', 'molecule_nat_lang')

    Returns:
        str: 画像出力ディレクトリのパス
    """
    learning_source_dir = check_learning_source_dir()
    image_dir = os.path.join(learning_source_dir, model_type, "image")
    os.makedirs(image_dir, exist_ok=True)
    return image_dir


def get_image_path(model_type: str, filename: str) -> str:
    """
    指定されたモデルタイプとファイル名の画像パスを取得

    Args:
        model_type (str): モデルタイプ
        filename (str): ファイル名（拡張子含む）

    Returns:
        str: 完全な画像ファイルパス
    """
    image_dir = get_image_output_dir(model_type)
    return os.path.join(image_dir, filename)


class ImageInfo(TypedDict):
    filename: str
    path: str
    size: int
    modified: float


def list_images_in_model_dir(model_type: str) -> list[ImageInfo]:
    """
    指定されたモデルタイプディレクトリ内の画像ファイル一覧を取得

    Args:
        model_type (str): モデルタイプ

    Returns:
        list: 画像ファイル情報のリスト [{'filename': str, 'path': str, 'size': int, 'modified': float}]
    """
    image_dir = get_image_output_dir(model_type)

    if not os.path.exists(image_dir):
        return []

    images: list[ImageInfo] = []
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"}

    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            filepath = os.path.join(image_dir, filename)
            try:
                stat = os.stat(filepath)
                images.append({"filename": filename, "path": filepath, "size": stat.st_size, "modified": stat.st_mtime})
            except OSError:
                continue

    # 更新日時でソート（新しい順）
    images.sort(key=lambda x: float(x["modified"]), reverse=True)
    return images


def migrate_legacy_images():
    """
    既存のassets/imgディレクトリから新しい構造に画像を移行
    """
    legacy_dir = os.path.join("assets", "img")

    if not os.path.exists(legacy_dir):
        return

    # 画像ファイルのマッピング（ファイル名パターンからモデルタイプを判定）
    model_mappings = {
        "protein_sequence": ["protein_sequence_"],
        "genome_sequence": ["genome_sequence_"],
        "compounds": ["compounds_"],
        "rna": ["rna_"],
        "molecule_nat_lang": ["molecule_nat_lang_"],
    }

    for filename in os.listdir(legacy_dir):
        if not any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"]):
            continue

        # ファイル名からモデルタイプを判定
        model_type = None
        for mt, patterns in model_mappings.items():
            if any(pattern in filename for pattern in patterns):
                model_type = mt
                break

        if model_type:
            source_path = os.path.join(legacy_dir, filename)
            dest_path = get_image_path(model_type, filename)

            # ファイルをコピー（既存ファイルがない場合のみ）
            if not os.path.exists(dest_path):
                try:
                    import shutil

                    shutil.copy2(source_path, dest_path)
                    print(f"Migrated: {source_path} -> {dest_path}")
                except Exception as e:
                    print(f"Failed to migrate {source_path}: {e}")


if __name__ == "__main__":
    # テスト実行
    print("Testing image manager...")

    # 各モデルタイプのディレクトリ作成テスト
    model_types = ["protein_sequence", "genome_sequence", "compounds", "rna", "molecule_nat_lang"]

    for model_type in model_types:
        image_dir = get_image_output_dir(model_type)
        print(f"Image directory for {model_type}: {image_dir}")

        # 画像一覧取得テスト
        images = list_images_in_model_dir(model_type)
        print(f"Images in {model_type}: {len(images)} files")

    # レガシー画像の移行テスト
    migrate_legacy_images()
