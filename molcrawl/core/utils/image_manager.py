#!/usr/bin/env python3
"""
Image management utility

Utility for unified management of images resulting from dataset preparation and verification
"""

import os
from typing import TypedDict

from molcrawl.core.utils.environment_check import check_learning_source_dir


def get_image_output_dir(model_type: str) -> str:
    """
    Get image output directory for specified model type

        Args:
    model_type (str): Model type ('protein_sequence', 'genome_sequence', 'compounds', 'rna', 'molecule_nat_lang')

        Returns:
    str: path of image output directory
    """
    learning_source_dir = check_learning_source_dir()
    image_dir = os.path.join(learning_source_dir, model_type, "image")
    os.makedirs(image_dir, exist_ok=True)
    return image_dir


def get_image_path(model_type: str, filename: str) -> str:
    """
    Get image path for specified model type and file name

        Args:
    model_type (str): Model type
    filename (str): File name (including extension)

        Returns:
    str: complete image file path
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
    Get a list of image files in the specified model type directory

        Args:
    model_type (str): Model type

        Returns:
    list: List of image file information [{'filename': str, 'path': str, 'size': int, 'modified': float}]
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

    # Sort by update date (newest first)
    images.sort(key=lambda x: float(x["modified"]), reverse=True)
    return images


def migrate_legacy_images():
    """
    Migrate images from existing assets/img directory to new structure
    """
    legacy_dir = os.path.join("assets", "img")

    if not os.path.exists(legacy_dir):
        return

    # Image file mapping (determine model type from file name pattern)
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

        # Determine model type from file name
        model_type = None
        for mt, patterns in model_mappings.items():
            if any(pattern in filename for pattern in patterns):
                model_type = mt
                break

        if model_type:
            source_path = os.path.join(legacy_dir, filename)
            dest_path = get_image_path(model_type, filename)

            # Copy file (only if there is no existing file)
            if not os.path.exists(dest_path):
                try:
                    import shutil

                    shutil.copy2(source_path, dest_path)
                    print(f"Migrated: {source_path} -> {dest_path}")
                except Exception as e:
                    print(f"Failed to migrate {source_path}: {e}")


if __name__ == "__main__":
    # test execution
    print("Testing image manager...")

    # Directory creation test for each model type
    model_types = ["protein_sequence", "genome_sequence", "compounds", "rna", "molecule_nat_lang"]

    for model_type in model_types:
        image_dir = get_image_output_dir(model_type)
        print(f"Image directory for {model_type}: {image_dir}")

        # Image list acquisition test
        images = list_images_in_model_dir(model_type)
        print(f"Images in {model_type}: {len(images)} files")

    # Legacy image migration test
    migrate_legacy_images()
