"""
This script will download and preprocess the cellxgene dataset.
There will be multiple directories generated in the output_dir provided in the configuration:

- download_dir: Raw archive file downloaded from the cellxgene database
- extract: h5ad file extracted from the archives
- parquet_files: parquet files containing tokenized gene and expression values

You can call this script with the following command:

    python scripts/preparation_script_rna.py assets/configs/rna.yaml
"""

import datetime
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from datasets.utils.logging import enable_progress_bar

# プロジェクトルートのsrcディレクトリをパスに追加

from src.config.paths import RNA_DATASET_DIR
from core.base import setup_logging
from rna.dataset.cellxgene.script.build_list import build_list
from rna.dataset.cellxgene.script.download import download
from rna.dataset.cellxgene.script.h5ad_to_loom import h5ad_to_loom
from rna.dataset.cellxgene.script.scgpt_tokenization import get_census_gene_vocab
from rna.dataset.tokenization import tokenize
from rna.utils.config import RnaConfig

logger = logging.getLogger(__name__)
enable_progress_bar()


def create_distribution_plot(data):
    """トークン長の分布をヒストグラムとして保存"""
    from utils.image_manager import get_image_path

    plt.hist(data["num_tokens"], bins=200)
    plt.xlabel("Length of tokenized dataset")
    plt.title("Distribution of tokenized lengths")
    plt.tight_layout()

    image_path = get_image_path("rna", "rna_tokenized_lengths_dist.png")
    plt.savefig(image_path)
    plt.close()
    logger.info(f"Saved distribution of tokenized dataset lengths to {image_path}")


# より詳細な遺伝子情報を含むTSVファイルの生成
def create_enhanced_gene_list(vocab, data, out_dir):
    """遺伝子の使用頻度や統計情報を含むTSVを作成"""

    # 各遺伝子の出現頻度を計算
    n = 0
    gene_counts = {}
    for info in data:
        for token_id in info["token"]:
            n += 1
            if n % 1_0000_0000 == 0:
                logger.info(f"{datetime.datetime.now()} Processed {n} items...")
            if token_id in gene_counts:
                gene_counts[token_id] += 1
            else:
                gene_counts[token_id] = 1

    # 遺伝子名でソート（またはIDでソート）
    inv_vocab = {v: k for k, v in vocab.items()}

    with open(out_dir / "gene_list_with_stats.tsv", "w") as f:
        f.write("gene_id\tgene_name\tcount\tfrequency\n")

        total_tokens = sum(gene_counts.values())
        for gene_id in sorted(inv_vocab.keys()):
            gene_name = inv_vocab[gene_id]
            count = gene_counts.get(gene_id, 0)
            frequency = count / total_tokens if total_tokens > 0 else 0
            f.write(f"{gene_id}\t{gene_name}\t{count}\t{frequency:.6f}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and reprocessing even if files exist",
    )
    args = parser.parse_args()
    cfg = RnaConfig.from_file(args.config).data_preparation

    setup_logging(Path(RNA_DATASET_DIR))

    # 各処理段階のマーカー（完了印）ファイル
    build_list_marker = Path(RNA_DATASET_DIR) / "build_list_complete.marker"
    download_marker = Path(RNA_DATASET_DIR) / "download_complete.marker"
    h5ad_to_loom_marker = Path(RNA_DATASET_DIR) / "h5ad_to_loom_complete.marker"
    tokenize_marker = Path(RNA_DATASET_DIR) / "tokenize_complete.marker"
    vocab_marker = Path(RNA_DATASET_DIR) / "gene_vocab.json"
    parquet_dir = Path(RNA_DATASET_DIR) / "parquet_files"

    # 進捗状況を表示
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

    # 1. Build list
    if not args.force and build_list_marker.exists():
        logger.info("Build list already completed. Skipping build_list step.")
    else:
        if args.force:
            logger.info("Force option specified. Rebuilding list...")
        logger.info("Building dataset list...")
        build_list(RNA_DATASET_DIR, cfg.census_version)
        build_list_marker.touch()
        logger.info("Build list completed.")

    # 2. Download
    if not args.force and download_marker.exists():
        logger.info("Download already completed. Skipping download step.")
    else:
        if args.force:
            logger.info("Force option specified. Re-downloading...")

        # Show estimated workload before starting
        from pathlib import Path
        from rna.dataset.cellxgene.script.download import divide_workload

        metadata_dir = Path(RNA_DATASET_DIR) / "metadata_preparation_dir"
        workload = divide_workload(metadata_dir, cfg.size_workload)

        logger.info("=" * 60)
        logger.info("Starting CellxGene dataset download")
        logger.info(f"Total download tasks: {len(workload)}")
        logger.info(f"Samples per task: {cfg.size_workload}")
        logger.info(f"Parallel workers: {cfg.num_worker}")
        logger.info(f"Estimated time: ~{len(workload) * 7 // cfg.num_worker // 60} minutes")
        logger.info("=" * 60)

        download(
            RNA_DATASET_DIR,
            cfg.census_version,
            cfg.num_worker,
            cfg.size_workload,
        )
        download_marker.touch()
        logger.info("=" * 60)
        logger.info("Download completed successfully!")
        logger.info("=" * 60)

    # 3. H5AD to Loom conversion
    if not args.force and h5ad_to_loom_marker.exists():
        logger.info("H5AD to Loom conversion already completed. Skipping conversion step.")
    else:
        if args.force:
            logger.info("Force option specified. Reconverting H5AD to Loom...")
        logger.info("Converting H5AD files to Loom format...")
        h5ad_to_loom(RNA_DATASET_DIR)
        h5ad_to_loom_marker.touch()
        logger.info("H5AD to Loom conversion completed.")

    # 4. Tokenization
    if not args.force and tokenize_marker.exists() and parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("Tokenization already completed. Skipping tokenization step.")
    else:
        if args.force:
            logger.info("Force option specified. Retokenizing...")
        logger.info("Tokenizing datasets...")
        tokenize(RNA_DATASET_DIR)
        tokenize_marker.touch()
        logger.info("Tokenization completed.")

    # 5. Vocabulary generation
    if not args.force and vocab_marker.exists():
        logger.info("Gene vocabulary already exists. Skipping vocabulary generation.")
    else:
        if args.force:
            logger.info("Force option specified. Regenerating vocabulary...")
        logger.info("Generating gene vocabulary...")
        vocab = get_census_gene_vocab(cfg.census_version)
        vocab.save_json(Path(RNA_DATASET_DIR) / "gene_vocab.json")
        logger.info("Gene vocabulary generation completed.")

    # 6. 統計情報の生成と保存
    logger.info("Generating statistics and final report...")

    try:
        # parquet データセットをロード（train split）
        data = load_dataset(
            "parquet",
            data_dir=str(Path(RNA_DATASET_DIR) / "parquet_files"),
            split="train",
            cache_dir=str(Path(RNA_DATASET_DIR) / "hf_cache"),
        )

        logger.info(f"Number of sequence: {len(data)}")

        vocab_file = Path(RNA_DATASET_DIR) / "gene_vocab.json"
        with open(vocab_file) as file:
            vocab = json.load(file)
        logger.info(f"Size of the vocabulary: {len(vocab)}")

        logger.info(f"Number of tokens: {sum(data['token_count'])}")

        # 分布プロット（必要に応じて再生成）
        from utils.image_manager import get_image_path

        plot_file = Path(get_image_path("rna", "rna_tokenized_lengths_dist.png"))
        if args.force or not plot_file.exists():
            if args.force:
                logger.info("Force option specified. Regenerating distribution plot...")
            logger.info("Creating distribution plot...")
            create_distribution_plot(data)
        else:
            logger.info("Distribution plot already exists. Skipping plot generation.")

        # === 追加: 統計と遺伝子リストを保存 ===
        out_dir = Path(RNA_DATASET_DIR)

        stats = {
            "num_sequences": len(data),
            "vocab_size": len(vocab),
            "num_tokens": int(sum(data["token_count"])),
        }
        with open(out_dir / "rna_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # token_id 順で並べた遺伝子リストを TSV で保存
        inv_vocab = sorted(vocab.items(), key=lambda x: x[1])  # (gene, id)
        with open(out_dir / "gene_list_with_id.tsv", "w") as f:
            for g, i in inv_vocab:
                f.write(f"{i}\t{g}\n")

        logger.info(f"Saved dataset stats and gene list (with IDs) in {out_dir}")

        # より詳細な遺伝子情報を含むTSVファイルの生成
        create_enhanced_gene_list(vocab, data, out_dir)

        logger.info("RNA dataset preparation completed successfully!")

    except Exception as e:
        logger.error(f"Failed to load or process final dataset: {e}")
        logger.error("Some processing steps may have failed. Please check logs and consider using --force option.")
        exit(1)
