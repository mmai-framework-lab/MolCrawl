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


def download_mol_instructions(save_path):
    """
    Download Mol-Instructions (Molecule-oriented Instructions subset) from
    Hugging Face Hub.

    The dataset uses task names as split keys instead of train/validation/test.
    Available splits: description_guided_molecule_design,
    forward_reaction_prediction, molecular_description_generation,
    property_prediction, reagent_prediction, retrosynthesis.

    Reference: Fang et al., "Mol-Instructions: A Large-Scale Biomolecular
    Instruction Dataset for Large Language Models", ICLR 2024.
    https://huggingface.co/datasets/zjunlp/Mol-Instructions

    Args:
        save_path: Path to save the dataset (will be created if not exists)
    """
    from datasets import load_dataset

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Downloading Mol-Instructions (Molecule-oriented) from Hugging Face Hub...")
        logger.info("Reference: https://huggingface.co/datasets/zjunlp/Mol-Instructions")

        cache_dir = save_path.parent / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        data = load_dataset(
            "zjunlp/Mol-Instructions",
            "Molecule-oriented Instructions",
            trust_remote_code=True,
            cache_dir=str(cache_dir),
        )

        logger.info(f"Dataset loaded. Splits (tasks): {list(data.keys())}")
        for split in data.keys():
            logger.info(f"  {split}: {len(data[split])} samples")

        logger.info(f"Saving dataset to {save_path}...")
        data.save_to_disk(str(save_path))

        logger.info("Mol-Instructions download completed successfully")

    except Exception as e:
        logger.error(f"Failed to download Mol-Instructions: {e}")
        logger.error("Possible solutions:")
        logger.error("- Check your internet connection")
        logger.error("- Run: huggingface-cli login")
        raise


if __name__ == "__main__":
    download_hf_dataset("src/molecule_nat_lang/assets/raw_data")
