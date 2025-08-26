"""
This script will download the cellxgene dataset.
There will be multiple directory generate in the output_dir provided in the configuration

- download_dir: Raw archive file downloaded from the cellxgene database
- extract: h5ad file extracted from the archives
- parquet_files: parquet files containing tokenized gene and expression values

You can call this script with the following command:

python scripts/preparation_script_rna.py assets/configs/rna.yaml
"""

from argparse import ArgumentParser
from pathlib import Path
import logging

import json
from datasets import load_dataset
import matplotlib.pyplot as plt

from rna.dataset.cellxgene.script.build_list import build_list
from rna.dataset.cellxgene.script.download import download
from rna.dataset.cellxgene.script.h5ad_to_loom import h5ad_to_loom
from rna.dataset.cellxgene.script.scgpt_tokenization import get_census_gene_vocab
from datasets.utils.logging import enable_progress_bar

from rna.dataset.tokenization import tokenize
from rna.utils.config import RnaConfig
from core.base import setup_logging

logger = logging.getLogger(__name__)
enable_progress_bar()

from config.paths import RNA_DATASET_DIR

def create_distribution_plot(data):
    plt.hist(data["num_tokens"], bins=200)
    plt.xlabel("Length of tokenized dataset")
    plt.title("Distribution of tokenized lengths")
    plt.tight_layout()
    plt.savefig("assets/img/rna_tokenized_lengths_dist.png")
    plt.close()
    logger.info(msg="Saved distribution of tokenized dataset lengths to assets/img/rna_tokenized_lengths_dist.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--force", action="store_true", help="Force re-download and reprocessing even if files exist")
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    setup_logging(Path(RNA_DATASET_DIR))

    # 各処理段階のマーカーファイルパス
    build_list_marker = Path(RNA_DATASET_DIR) / "build_list_complete.marker"
    download_marker = Path(RNA_DATASET_DIR) / "download_complete.marker"
    h5ad_to_loom_marker = Path(RNA_DATASET_DIR) / "h5ad_to_loom_complete.marker"
    tokenize_marker = Path(RNA_DATASET_DIR) / "tokenize_complete.marker"
    vocab_marker = Path(RNA_DATASET_DIR) / "gene_vocab.json"
    parquet_dir = Path(RNA_DATASET_DIR) / "parquet_files"

    # 進捗状況の確認
    logger.info("=== RNA Dataset Preparation Progress ===")
    steps_completed = 0
    total_steps = 5
    
    if build_list_marker.exists():
        logger.info("✓ Step 1/5: Build list - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 1/5: Build list - PENDING")
        
    if download_marker.exists():
        logger.info("✓ Step 2/5: Download - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 2/5: Download - PENDING")
        
    if h5ad_to_loom_marker.exists():
        logger.info("✓ Step 3/5: H5AD to Loom conversion - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 3/5: H5AD to Loom conversion - PENDING")
        
    if tokenize_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("✓ Step 4/5: Tokenization - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 4/5: Tokenization - PENDING")
        
    if vocab_marker.exists():
        logger.info("✓ Step 5/5: Vocabulary generation - COMPLETED")
        steps_completed += 1
    else:
        logger.info("⏳ Step 5/5: Vocabulary generation - PENDING")
        
    logger.info(f"Progress: {steps_completed}/{total_steps} steps completed")
    
    if steps_completed == total_steps and not args.force:
        logger.info("All processing steps are already completed!")
        logger.info("Use --force option if you want to reprocess everything.")
        
    logger.info("=========================================")

    # 1. Build list処理
    if not args.force and build_list_marker.exists():
        logger.info("Build list already completed. Skipping build_list step.")
        logger.info("Use --force option to rebuild.")
    else:
        if args.force:
            logger.info("Force option specified. Rebuilding list...")
        logger.info("Building dataset list...")
        build_list(RNA_DATASET_DIR, cfg.census_version)
        build_list_marker.touch()
        logger.info("Build list completed.")

    # 2. Download処理
    if not args.force and download_marker.exists():
        logger.info("Download already completed. Skipping download step.")
        logger.info("Use --force option to re-download.")
    else:
        if args.force:
            logger.info("Force option specified. Re-downloading...")
        logger.info("Downloading datasets...")
        download(RNA_DATASET_DIR, cfg.census_version, cfg.num_worker, cfg.size_workload)
        download_marker.touch()
        logger.info("Download completed.")

    # 3. H5AD to Loom conversion
    if not args.force and h5ad_to_loom_marker.exists():
        logger.info("H5AD to Loom conversion already completed. Skipping conversion step.")
        logger.info("Use --force option to reconvert.")
    else:
        if args.force:
            logger.info("Force option specified. Reconverting H5AD to Loom...")
        logger.info("Converting H5AD files to Loom format...")
        h5ad_to_loom(RNA_DATASET_DIR)
        h5ad_to_loom_marker.touch()
        logger.info("H5AD to Loom conversion completed.")

    # 4. Tokenization処理
    if not args.force and tokenize_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("Tokenization already completed. Skipping tokenization step.")
        logger.info("Use --force option to retokenize.")
    else:
        if args.force:
            logger.info("Force option specified. Retokenizing...")
        logger.info("Tokenizing datasets...")
        tokenize(RNA_DATASET_DIR)
        tokenize_marker.touch()
        logger.info("Tokenization completed.")

    # 5. Vocabulary生成
    if not args.force and vocab_marker.exists():
        logger.info("Gene vocabulary already exists. Skipping vocabulary generation.")
        logger.info("Use --force option to regenerate vocabulary.")
    else:
        if args.force:
            logger.info("Force option specified. Regenerating vocabulary...")
        logger.info("Generating gene vocabulary...")
        vocab = get_census_gene_vocab(cfg.census_version)
        vocab.save_json(Path(RNA_DATASET_DIR) / "gene_vocab.json")
        logger.info("Gene vocabulary generation completed.")

    # 6. 統計情報の生成とレポート
    logger.info("Generating statistics and final report...")
    
    # データセットの読み込み確認
    try:
        data = load_dataset(
            "parquet",
            data_dir=str(Path(RNA_DATASET_DIR) / "parquet_files"),
            split="train",
            cache_dir=str(Path(RNA_DATASET_DIR) / "hf_cache"),
        )
        
        logger.info(f"Number of sequence: {len(data)}")
        
        with open(Path(RNA_DATASET_DIR) / "gene_vocab.json") as file:
            vocab = json.load(file)
        logger.info(f"Size of the vocabulary: {len(vocab)}")

        logger.info(f"Number of tokens: {sum(data['token_count'])}")
        
        # 分布プロットの生成（forceオプションまたはプロットが存在しない場合のみ）
        plot_file = Path("assets/img/rna_tokenized_lengths_dist.png")
        if args.force or not plot_file.exists():
            if args.force:
                logger.info("Force option specified. Regenerating distribution plot...")
            logger.info("Creating distribution plot...")
            create_distribution_plot(data)
        else:
            logger.info("Distribution plot already exists. Skipping plot generation.")
            logger.info("Use --force option to regenerate plot.")
            
        logger.info("RNA dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load or process final dataset: {e}")
        logger.error("Some processing steps may have failed. Please check the logs and consider using --force option.")
        exit(1)
