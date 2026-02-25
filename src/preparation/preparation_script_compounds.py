#!/usr/bin/env python3
"""
化合物データセット準備スクリプト

個別データセット処理により、部分的なダウンロードに対応します。

使用例:
    # 全データセットを処理
    python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml

    # 特定のデータセットのみダウンロード
    python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml \
        --download-only --datasets zinc20 opv

    # 処理とトークナイズのみ（ダウンロード済みの場合）
    python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml \
        --skip-download

    # 強制再処理
    python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml --force
"""

import logging
import logging.config
import os
from argparse import ArgumentParser
from pathlib import Path

from compounds.dataset.dataset_config import CompoundDatasetType, get_all_dataset_types
from compounds.dataset.hf_converter import convert_all_tokenized_datasets
from compounds.dataset.processor import process_all_available_datasets
from compounds.dataset.tokenizer import (
    compute_tokenization_statistics,
    tokenize_all_processed_datasets,
)
from compounds.utils.config import CompoundConfig
from compounds.utils.general import (
    download_llamol_datasets,
    download_opv,
    download_zinc20,
)
from config.paths import COMPOUNDS_DIR
from core.base import setup_logging

logger = logging.getLogger(__name__)


def download_datasets_individually(cfg, compounds_dir, dataset_types, force=False):
    """
    個別にデータセットをダウンロード

    Args:
        cfg: 設定オブジェクト
        compounds_dir: compoundsディレクトリ
        dataset_types: ダウンロードするデータセット種別のリスト
        force: 強制再ダウンロードフラグ
    """
    data_dir = os.path.join(compounds_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    for dataset_type in dataset_types:
        marker_file = Path(data_dir) / f"{dataset_type.value}_download.marker"

        if not force and marker_file.exists():
            logger.info(f"✓ {dataset_type.value}: Already downloaded, skipping")
            continue

        logger.info(f"📥 Downloading {dataset_type.value}...")

        try:
            if dataset_type == CompoundDatasetType.ZINC20:
                download_zinc20(compounds_dir)
            elif dataset_type == CompoundDatasetType.OPV:
                download_opv(compounds_dir)
            elif dataset_type in [
                CompoundDatasetType.PC9_GAP,
                CompoundDatasetType.ZINC_QM9,
                CompoundDatasetType.REDDB,
                CompoundDatasetType.CHEMBL,
                CompoundDatasetType.PUBCHEMQC_2017,
                CompoundDatasetType.PUBCHEMQC_2020,
            ]:
                # LlaMolデータセットはまとめてダウンロード
                llamol_marker = Path(data_dir) / "llamol_download.marker"
                if not force and llamol_marker.exists():
                    logger.info("✓ LlaMol datasets: Already downloaded, skipping")
                    continue
                download_llamol_datasets(compounds_dir)
                llamol_marker.touch()
                # 個別のマーカーも作成
                for dt in [
                    CompoundDatasetType.PC9_GAP,
                    CompoundDatasetType.ZINC_QM9,
                    CompoundDatasetType.REDDB,
                    CompoundDatasetType.CHEMBL,
                    CompoundDatasetType.PUBCHEMQC_2017,
                    CompoundDatasetType.PUBCHEMQC_2020,
                ]:
                    (Path(data_dir) / f"{dt.value}_download.marker").touch()
                break  # LlaMolは一度だけダウンロード
            elif dataset_type == CompoundDatasetType.GUACAMOL:
                logger.info(f"⚠ {dataset_type.value}: Please download manually from GuacaMol benchmark")
                continue
            else:
                logger.warning(f"⚠ Unknown dataset type: {dataset_type.value}")
                continue

            marker_file.touch()
            logger.info(f"✓ {dataset_type.value}: Download completed")

        except Exception as e:
            logger.error(f"✗ {dataset_type.value}: Download failed - {e}")


def main():
    """メイン実行関数"""
    parser = ArgumentParser(description="化合物データセット準備スクリプト")
    parser.add_argument("config", help="設定ファイルのパス")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[dt.value for dt in get_all_dataset_types()],
        help="処理するデータセット（指定しない場合は利用可能な全て）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="強制再処理（既存ファイルを上書き）",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="ダウンロードのみ実行",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="ダウンロードをスキップ",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="処理（物性計算）をスキップ",
    )
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="トークナイズをスキップ",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="HuggingFace形式への変換をスキップ",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="統計計算・可視化をスキップ",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="並列処理のプロセス数（物性計算用、デフォルト: 16）",
    )
    parser.add_argument(
        "--tokenization-processes",
        type=int,
        default=2,
        help="トークナイズの並列処理数（デフォルト: 2）",
    )

    args = parser.parse_args()

    # 設定の読み込み
    cfg = CompoundConfig.from_file(args.config).data_preparation
    compounds_dir = COMPOUNDS_DIR
    os.makedirs(compounds_dir, exist_ok=True)

    # ロギングのセットアップ
    setup_logging(compounds_dir + "/compounds_logs")

    logger.info("=" * 70)
    logger.info("化合物データセット準備スクリプト（改訂版）")
    logger.info("=" * 70)
    logger.info(f"Compounds directory: {compounds_dir}")

    # 処理するデータセットを決定
    if args.datasets:
        dataset_types = [CompoundDatasetType(dt) for dt in args.datasets]
        logger.info(f"Target datasets: {[dt.value for dt in dataset_types]}")
    else:
        dataset_types = None
        logger.info("Target datasets: All available")

    # ステップ1: ダウンロード
    if not args.skip_download and not args.skip_process and not args.skip_tokenize and not args.skip_convert:
        run_download = True
    elif args.download_only:
        run_download = True
    else:
        run_download = not args.skip_download

    if run_download:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: データセットダウンロード")
        logger.info("=" * 70)

        if dataset_types:
            download_datasets_individually(cfg, compounds_dir, dataset_types, args.force)
        else:
            # 全データセットをダウンロード
            all_types = [dt for dt in get_all_dataset_types() if dt != CompoundDatasetType.GUACAMOL]
            download_datasets_individually(cfg, compounds_dir, all_types, args.force)

    if args.download_only:
        logger.info("\n✅ ダウンロードのみ完了")
        return

    # ステップ2: 処理（物性計算）
    if not args.skip_process:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: データセット処理（物性計算）")
        logger.info("=" * 70)

        processed = process_all_available_datasets(
            Path(compounds_dir),
            dataset_types=[CompoundDatasetType(dt) for dt in args.datasets] if args.datasets else None,
            force=args.force,
            num_processes=args.num_processes,
        )

        logger.info(f"\n✓ {len(processed)} datasets processed")

    # ステップ3: トークナイズ
    if not args.skip_tokenize:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: トークナイズ")
        logger.info("=" * 70)

        tokenized = tokenize_all_processed_datasets(
            Path(compounds_dir),
            cfg.vocab_path,
            cfg.max_length,
            dataset_types=[CompoundDatasetType(dt) for dt in args.datasets] if args.datasets else None,
            force=args.force,
            num_processes=args.tokenization_processes,
        )

        logger.info(f"\n✓ {len(tokenized)} datasets tokenized")

    # ステップ4: HuggingFace形式に変換
    if not args.skip_convert:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: HuggingFace Dataset形式に変換")
        logger.info("=" * 70)

        converted = convert_all_tokenized_datasets(
            Path(compounds_dir),
            dataset_types=[CompoundDatasetType(dt) for dt in args.datasets] if args.datasets else None,
            train_ratio=0.9,
            valid_ratio=0.05,
            test_ratio=0.05,
            force=args.force,
        )

        logger.info(f"\n✓ {len(converted)} datasets converted")

    # ステップ5: 統計計算・可視化
    if not args.skip_stats:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: 統計計算・可視化")
        logger.info("=" * 70)

        stats = compute_tokenization_statistics(
            Path(compounds_dir),
            dataset_types=[CompoundDatasetType(dt) for dt in args.datasets] if args.datasets else None,
            force=args.force,
        )

        logger.info(f"\n✓ {len(stats)} datasets statistics computed")

    logger.info("\n" + "=" * 70)
    logger.info("✅ 全ての処理が完了しました")
    logger.info("=" * 70)
    logger.info(f"Output directory: {compounds_dir}")
    logger.info(f"  - Processed data: {compounds_dir}/processed/")
    logger.info(f"  - Tokenized data: {compounds_dir}/tokenized/")
    logger.info(f"  - HuggingFace datasets: {compounds_dir}/hf_datasets/")


if __name__ == "__main__":
    main()
