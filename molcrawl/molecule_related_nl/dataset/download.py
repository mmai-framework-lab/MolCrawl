import os
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


def download_hf_dataset(save_path):
    from datasets import load_dataset

    """
    Download SMolInstruct dataset from Hugging Face Hub

    Args:
        save_path: Path to save the dataset
    """
    os.path.exists(save_path) or os.makedirs(save_path)

    try:
        logger.info("Downloading SMolInstruct dataset from Hugging Face Hub...")
        logger.info("This may take a while as the dataset is large...")

        # Set up cache directory
        cache_dir = Path(save_path).parent / "hf_cache"
        os.makedirs(cache_dir, exist_ok=True)

        # Load the dataset with trust_remote_code=True
        # Note: download_mode default is 'reuse_dataset_if_exists'
        data = load_dataset("osunlp/SMolInstruct", trust_remote_code=True, cache_dir=str(cache_dir))

        logger.info(f"Dataset loaded successfully. Found splits: {list(data.keys())}")

        # Log dataset information
        for split in data.keys():
            logger.info(f"  {split}: {len(data[split])} samples")

        # Save the entire DatasetDict to disk
        logger.info(f"Saving dataset to {save_path}...")
        data.save_to_disk(str(save_path))

        logger.info("Dataset download and save completed successfully")

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.error("This may be due to:")
        logger.error("1. Network connection issues")
        logger.error("2. Hugging Face Hub authentication (some datasets require login)")
        logger.error("3. Dataset format changes")
        logger.error("\nPossible solutions:")
        logger.error("- Check your internet connection")
        logger.error("- Run: huggingface-cli login")
        logger.error("- Try clearing the cache and re-running with --force")
        raise


if __name__ == "__main__":
    download_hf_dataset("src/molecule_related_nl/assets/raw_data")
