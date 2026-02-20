#!/usr/bin/env python3
"""
ProteinGymデータセットのダウンロードと前処理ユーティリティ

このスクリプトは、ProteinGymデータセットをダウンロードし、
評価用に適切な形式に前処理します。
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# 共通モジュールを追加
from src.utils.environment_check import check_learning_source_dir

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProteinGymDataDownloader:
    """ProteinGymデータセットのダウンロードと前処理クラス"""

    # ProteinGym v1.3データセットの公式URL
    PROTEINGYM_URLS = {
        # DMS (Deep Mutational Scanning) データ - メイン評価用
        "substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip",
        "indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_indels.zip",
        # 参照ファイル - アッセイメタデータ
        "reference_substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv",
        "reference_indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_indels.csv",
        # 臨床変異データ - 補完評価用
        "clinical_substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_ProteinGym_substitutions.zip",
        "clinical_indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_ProteinGym_indels.zip",
        "clinical_reference_substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_substitutions.csv",
        "clinical_reference_indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_indels.csv",
        # 生データ（必要に応じて）
        "raw_substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/substitutions_raw_DMS.zip",
        "raw_indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/indels_raw_DMS.zip",
        # 多重配列アライメント（高度な分析用）
        "msa_dms": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip",
        "msa_clinical": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_msa_files.zip",
        # タンパク質構造（構造ベース分析用）
        "structures": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/ProteinGym_AF2_structures.zip",
    }

    def __init__(self, data_dir="./proteingym_data"):
        """
        初期化

        Args:
            data_dir (str): データ保存ディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filename=None, force_download=False):
        """
        ファイルをダウンロード

        Args:
            url (str): ダウンロードURL
            filename (str): 保存ファイル名（Noneの場合はURLから推定）
            force_download (bool): 既存ファイルを上書きするか

        Returns:
            str: ダウンロードされたファイルのパス
        """
        if filename is None:
            filename = os.path.basename(urlparse(url).path)

        filepath = self.data_dir / filename

        if filepath.exists() and not force_download:
            logger.info(f"File already exists: {filepath}")
            return str(filepath)

        logger.info(f"Downloading {url} to {filepath}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(filepath, "wb") as f,
                tqdm(
                    desc=filename,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Downloaded successfully: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            raise

    def extract_zip(self, zip_path, extract_dir=None):
        """
        ZIPファイルを展開

        Args:
            zip_path (str): ZIPファイルのパス
            extract_dir (str): 展開先ディレクトリ（Noneの場合は同じディレクトリ）

        Returns:
            str: 展開先ディレクトリ
        """
        zip_path = Path(zip_path)

        if extract_dir is None:
            extract_dir = zip_path.parent / zip_path.stem
        else:
            extract_dir = Path(extract_dir)

        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting {zip_path} to {extract_dir}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        logger.info(f"Extracted successfully to {extract_dir}")
        return str(extract_dir)

    def download_proteingym_data(self, data_type="substitutions", force_download=False):
        """
        ProteinGymデータセットをダウンロード

        Args:
            data_type (str): データタイプ
                          - 'substitutions': DMS単一置換データ（推奨）
                          - 'indels': DMS挿入・欠失データ
                          - 'clinical_substitutions': 臨床変異置換データ
                          - 'clinical_indels': 臨床変異挿入・欠失データ
                          - 'reference_substitutions': DMS置換参照ファイル
                          - 'reference_indels': DMS挿入・欠失参照ファイル
                          - 'msa_dms': 多重配列アライメント（DMS）
                          - 'structures': タンパク質構造データ
            force_download (bool): 強制ダウンロード

        Returns:
            str: ダウンロードされたファイル/ディレクトリのパス
        """
        if data_type not in self.PROTEINGYM_URLS:
            available_types = list(self.PROTEINGYM_URLS.keys())
            raise ValueError(f"Invalid data_type: {data_type}. Choose from {available_types}")

        url = self.PROTEINGYM_URLS[data_type]
        downloaded_file = self.download_file(url, force_download=force_download)

        # ZIPファイルの場合は展開
        if downloaded_file.endswith(".zip"):
            extracted_dir = self.extract_zip(downloaded_file)
            return extracted_dir
        else:
            return downloaded_file

    def load_reference_file(self, reference_path=None, data_type="substitutions"):
        """
        ProteinGym参照ファイルを読み込み

        Args:
            reference_path (str): 参照ファイルのパス（Noneの場合は自動ダウンロード）
            data_type (str): データタイプ ('substitutions' or 'indels')

        Returns:
            pd.DataFrame: 参照データ
        """
        if reference_path is None:
            reference_key = f"reference_{data_type}"
            if reference_key not in self.PROTEINGYM_URLS:
                reference_key = "reference_substitutions"  # デフォルト
            reference_path = self.download_proteingym_data(reference_key)

        logger.info(f"Loading reference file: {reference_path}")
        df = pd.read_csv(reference_path)
        logger.info(f"Loaded {len(df)} assays from reference file")
        logger.info(f"Available columns: {list(df.columns)}")

        return df

    def prepare_evaluation_data(
        self,
        assay_id,
        data_dir=None,
        max_variants=None,
        balanced_sampling=False,
        positive_samples=1000,
        negative_samples=1000,
        score_threshold=None,
    ):
        """
        特定のアッセイの評価データを準備

        Args:
            assay_id (str): アッセイID
            data_dir (str): データディレクトリ（Noneの場合は自動ダウンロード）
            max_variants (int): 最大変異数（制限しない場合はNone）
            balanced_sampling (bool): バランスサンプリングを使用するか
            positive_samples (int): 陽性サンプル数（balanced_sampling=Trueの場合）
            negative_samples (int): 陰性サンプル数（balanced_sampling=Trueの場合）
            score_threshold (float): 陽性/陰性の閾値（Noneの場合は中央値を使用）

        Returns:
            pd.DataFrame: 評価用データ
        """
        if data_dir is None:
            data_dir = self.download_proteingym_data("substitutions")

        # アッセイファイルを探す
        assay_file = None
        data_path = Path(data_dir)

        for file_path in data_path.rglob(f"{assay_id}.csv"):
            assay_file = file_path
            break

        if assay_file is None:
            raise FileNotFoundError(f"Assay file not found for ID: {assay_id}")

        logger.info(f"Loading assay data: {assay_file}")
        df = pd.read_csv(assay_file)

        # 必要なカラムをチェック
        required_columns = ["mutant", "mutated_sequence", "DMS_score"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # データのクリーニング
        df = df.dropna(subset=["DMS_score"])
        original_size = len(df)

        # バランスサンプリングの実行
        if balanced_sampling:
            df = self._balanced_sampling(df, positive_samples, negative_samples, score_threshold)
            logger.info(f"Applied balanced sampling: {original_size} → {len(df)} variants")
        elif max_variants and len(df) > max_variants:
            # 従来のランダムサンプリング
            logger.info(f"Limiting to {max_variants} variants (from {len(df)})")
            df = df.sample(n=max_variants, random_state=42)

        logger.info(f"Prepared {len(df)} variants for evaluation")
        logger.info(f"DMS score range: {df['DMS_score'].min():.3f} to {df['DMS_score'].max():.3f}")

        return df

    def create_test_dataset(self, output_file, n_variants=100):
        """
        テスト用の小さなデータセットを作成

        Args:
            output_file (str): 出力ファイルパス
            n_variants (int): 変異数
        """
        logger.info(f"Creating test dataset with {n_variants} variants")

        # サンプルタンパク質配列
        base_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAAFRVTLNEKLATWTEESS"

        # ランダムな変異を生成
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        data = []

        np.random.seed(42)

        for i in range(n_variants):
            if i == 0:
                # 野生型
                mutant = "WT"
                mutated_seq = base_sequence
                score = 1.0
            else:
                # ランダム変異
                pos = np.random.randint(1, len(base_sequence) + 1)
                orig_aa = base_sequence[pos - 1]
                mut_aa = np.random.choice([aa for aa in amino_acids if aa != orig_aa])

                mutant = f"{orig_aa}{pos}{mut_aa}"
                mutated_seq = base_sequence[: pos - 1] + mut_aa + base_sequence[pos:]

                # ランダムなスコア（より現実的な分布）
                score = np.random.beta(2, 5)  # 0に偏った分布

            data.append(
                {
                    "mutant": mutant,
                    "mutated_sequence": mutated_seq,
                    "DMS_score": score,
                    "protein_name": "TEST_PROTEIN",
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        logger.info(f"Test dataset created: {output_file}")
        logger.info(f"Score statistics: mean={df['DMS_score'].mean():.3f}, std={df['DMS_score'].std():.3f}")

    def _balanced_sampling(self, df, positive_samples=1000, negative_samples=1000, score_threshold=None):
        """
        陽性と陰性のサンプルをバランス良く抽出

        Args:
            df (pd.DataFrame): 元データフレーム
            positive_samples (int): 陽性サンプル数
            negative_samples (int): 陰性サンプル数
            score_threshold (float): 陽性/陰性の閾値（Noneの場合は中央値を使用）

        Returns:
            pd.DataFrame: バランス抽出されたデータフレーム
        """
        if score_threshold is None:
            score_threshold = df["DMS_score"].median()
            logger.info(f"Using median as threshold: {score_threshold:.3f}")
        else:
            logger.info(f"Using specified threshold: {score_threshold:.3f}")

        # 陽性・陰性にラベル分け
        positive_df = df[df["DMS_score"] >= score_threshold].copy()
        negative_df = df[df["DMS_score"] < score_threshold].copy()

        logger.info(f"Original distribution: {len(positive_df)} positive, {len(negative_df)} negative")

        # 各クラスからランダムサンプリング
        sampled_dfs = []

        # 陽性サンプルの抽出
        if len(positive_df) >= positive_samples:
            positive_sampled = positive_df.sample(n=positive_samples, random_state=42)
            logger.info(f"Sampled {positive_samples} positive samples from {len(positive_df)} available")
        else:
            positive_sampled = positive_df.copy()
            logger.warning(f"Only {len(positive_df)} positive samples available (requested {positive_samples})")

        sampled_dfs.append(positive_sampled)

        # 陰性サンプルの抽出
        if len(negative_df) >= negative_samples:
            negative_sampled = negative_df.sample(n=negative_samples, random_state=42)
            logger.info(f"Sampled {negative_samples} negative samples from {len(negative_df)} available")
        else:
            negative_sampled = negative_df.copy()
            logger.warning(f"Only {len(negative_df)} negative samples available (requested {negative_samples})")

        sampled_dfs.append(negative_sampled)

        # 結合してシャッフル
        balanced_df = pd.concat(sampled_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # バランス情報をログ出力
        final_positive = len(balanced_df[balanced_df["DMS_score"] >= score_threshold])
        final_negative = len(balanced_df[balanced_df["DMS_score"] < score_threshold])
        logger.info(f"Final balanced dataset: {final_positive} positive, {final_negative} negative")

        return balanced_df

    def prepare_multiple_assays_balanced(
        self,
        assay_ids,
        data_dir=None,
        positive_samples=1000,
        negative_samples=1000,
        score_threshold=None,
        output_dir=None,
    ):
        """
        複数のアッセイからバランス抽出したデータを準備

        Args:
            assay_ids (list): アッセイIDのリスト
            data_dir (str): データディレクトリ
            positive_samples (int): アッセイあたりの陽性サンプル数
            negative_samples (int): アッセイあたりの陰性サンプル数
            score_threshold (float): 陽性/陰性の閾値
            output_dir (str): 出力ディレクトリ

        Returns:
            dict: {assay_id: dataframe} の辞書
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        balanced_datasets = {}

        for assay_id in assay_ids:
            try:
                logger.info(f"Processing assay: {assay_id}")
                balanced_df = self.prepare_evaluation_data(
                    assay_id=assay_id,
                    data_dir=data_dir,
                    balanced_sampling=True,
                    positive_samples=positive_samples,
                    negative_samples=negative_samples,
                    score_threshold=score_threshold,
                )

                balanced_datasets[assay_id] = balanced_df

                # 個別ファイルに保存
                if output_dir:
                    output_file = output_path / f"{assay_id}_balanced.csv"
                    balanced_df.to_csv(output_file, index=False)
                    logger.info(f"Saved balanced dataset: {output_file}")

            except Exception as e:
                logger.error(f"Failed to process assay {assay_id}: {e}")
                continue

        return balanced_datasets

    def setup_test_data_from_metadata(
        self,
        metadata_file=None,
        positive_samples=1000,
        negative_samples=1000,
        test_assay_count=5,
    ):
        """
        メタデータファイルからテスト用のバランスデータセットを作成

        Args:
            metadata_file (str): メタデータファイルのパス
            positive_samples (int): 陽性サンプル数
            negative_samples (int): 陰性サンプル数
            test_assay_count (int): 作成するテストアッセイ数

        Returns:
            dict: 作成されたテストデータセットの情報
        """
        if metadata_file is None:
            # デフォルトのメタデータファイルを探す
            possible_paths = [
                Path(self.data_dir) / "DMS_substitutions.csv",
                Path(self.data_dir) / "reference_substitutions.csv",
            ]

            metadata_file = None
            for path in possible_paths:
                if path.exists():
                    metadata_file = path
                    break

            if metadata_file is None:
                raise FileNotFoundError("No metadata file found. Please download recommended datasets first.")

        logger.info(f"Setting up test data from metadata: {metadata_file}")

        # メタデータを読み込み
        metadata_df = pd.read_csv(metadata_file)
        logger.info(f"Found {len(metadata_df)} assays in metadata")

        # テスト用ディレクトリを作成
        test_data_dir = Path(self.data_dir) / "balanced_evaluation_data"
        test_data_dir.mkdir(exist_ok=True)

        # 上位のアッセイからテスト用を選択
        test_assays = metadata_df.head(test_assay_count)
        created_datasets = {}

        for _, row in test_assays.iterrows():
            assay_id = row["DMS_id"]
            target_sequence = row.get("target_seq", "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL")  # デフォルト配列

            logger.info(f"Creating test data for assay: {assay_id}")

            # サンプル変異データを生成
            test_data = self._generate_sample_mutations(
                assay_id=assay_id,
                target_seq=target_sequence,
                positive_samples=positive_samples,
                negative_samples=negative_samples,
            )

            # ファイルに保存
            output_file = test_data_dir / f"{assay_id}_balanced_evaluation_data.csv"
            test_data.to_csv(output_file, index=False)

            created_datasets[assay_id] = {
                "file": str(output_file),
                "total_samples": len(test_data),
                "positive_samples": len(test_data[test_data["DMS_score"] >= 0]),
                "negative_samples": len(test_data[test_data["DMS_score"] < 0]),
            }

            logger.info(f"Created test dataset: {output_file} ({len(test_data)} samples)")

        return created_datasets

    def _generate_sample_mutations(self, assay_id, target_seq, positive_samples, negative_samples):
        """
        特定のアッセイ用にサンプル変異データを生成

        Args:
            assay_id (str): アッセイID
            target_seq (str): 標的配列
            positive_samples (int): 陽性サンプル数
            negative_samples (int): 陰性サンプル数

        Returns:
            pd.DataFrame: 生成された変異データ
        """
        import random

        mutations = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # ランダムシードを設定（再現性のため）
        random.seed(42)
        np.random.seed(42)

        # 陽性サンプル生成（高いDMS_score: 0.5〜1.5）
        for _i in range(positive_samples):
            pos = random.randint(1, min(len(target_seq), 200))  # 配列長の制限
            if pos <= len(target_seq):
                orig_aa = target_seq[pos - 1]
                mut_aa = random.choice([aa for aa in amino_acids if aa != orig_aa])

                # 変異配列を作成
                mut_sequence = target_seq[: pos - 1] + mut_aa + target_seq[pos:]
            else:
                orig_aa = "A"
                mut_aa = random.choice(amino_acids)
                mut_sequence = target_seq + mut_aa

            mutations.append(
                {
                    "mutant": f"{orig_aa}{pos}{mut_aa}",
                    "mutated_sequence": mut_sequence,
                    "target_seq": target_seq,
                    "DMS_score": np.random.uniform(0.5, 1.5),  # 陽性スコア
                    "protein_name": assay_id,
                    "DMS_id": assay_id,
                }
            )

        # 陰性サンプル生成（低いDMS_score: -1.5〜0.0）
        for _i in range(negative_samples):
            pos = random.randint(1, min(len(target_seq), 200))
            if pos <= len(target_seq):
                orig_aa = target_seq[pos - 1]
                mut_aa = random.choice([aa for aa in amino_acids if aa != orig_aa])
                mut_sequence = target_seq[: pos - 1] + mut_aa + target_seq[pos:]
            else:
                orig_aa = "A"
                mut_aa = random.choice(amino_acids)
                mut_sequence = target_seq + mut_aa

            mutations.append(
                {
                    "mutant": f"{orig_aa}{pos}{mut_aa}",
                    "mutated_sequence": mut_sequence,
                    "target_seq": target_seq,
                    "DMS_score": np.random.uniform(-1.5, 0.0),  # 陰性スコア
                    "protein_name": assay_id,
                    "DMS_id": assay_id,
                }
            )

        # データフレームを作成してシャッフル
        df = pd.DataFrame(mutations)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    def download_recommended_datasets(self, force_download=False):
        """
        protein_sequence評価に推奨されるデータセットをダウンロード

        Args:
            force_download (bool): 強制ダウンロード

        Returns:
            dict: ダウンロードされたファイルパスの辞書
        """
        logger.info("Downloading recommended datasets for protein_sequence evaluation...")

        recommended_datasets = [
            "substitutions",  # メイン評価用：DMS単一置換データ
            "reference_substitutions",  # アッセイメタデータ
            "clinical_substitutions",  # 補完評価用：臨床変異データ
            "clinical_reference_substitutions",  # 臨床変異メタデータ
        ]

        downloaded_paths = {}

        for dataset in recommended_datasets:
            try:
                logger.info(f"Downloading {dataset}...")
                path = self.download_proteingym_data(dataset, force_download=force_download)
                downloaded_paths[dataset] = path
                logger.info(f"✓ {dataset} downloaded to: {path}")
            except Exception as e:
                logger.warning(f"Failed to download {dataset}: {e}")
                downloaded_paths[dataset] = None

        return downloaded_paths

    def get_small_test_assays(self, reference_df, max_assays=5, max_variants_per_assay=500):
        """
        テスト用の小さなアッセイを選択

        Args:
            reference_df (pd.DataFrame): 参照データフレーム
            max_assays (int): 最大アッセイ数
            max_variants_per_assay (int): アッセイあたりの最大変異数

        Returns:
            list: 選択されたアッセイIDのリスト
        """
        # 変異数でフィルタリング
        if "DMS_total_number_mutants" in reference_df.columns:
            filtered_df = reference_df[
                (reference_df["DMS_total_number_mutants"] <= max_variants_per_assay)
                & (reference_df["DMS_total_number_mutants"] >= 50)  # 最小50変異
            ]
        else:
            filtered_df = reference_df

        # ランダムに選択
        if len(filtered_df) > max_assays:
            selected_df = filtered_df.sample(n=max_assays, random_state=42)
        else:
            selected_df = filtered_df

        assay_ids = selected_df["DMS_id"].tolist()

        logger.info(f"Selected {len(assay_ids)} test assays:")
        for assay_id in assay_ids:
            row = selected_df[selected_df["DMS_id"] == assay_id].iloc[0]
            n_variants = row.get("DMS_total_number_mutants", "Unknown")
            protein_name = row.get("UniProt_ID", "Unknown")
            logger.info(f"  - {assay_id}: {protein_name} ({n_variants} variants)")

        return assay_ids


def main():
    # LEARNING_SOURCE_DIRの設定
    learning_source_dir = check_learning_source_dir()

    parser = argparse.ArgumentParser(description="ProteinGym data downloader and preparation utility")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Data directory for ProteinGym datasets",
    )
    parser.add_argument(
        "--download",
        choices=[
            "substitutions",
            "indels",
            "clinical_substitutions",
            "clinical_indels",
            "reference_substitutions",
            "reference_indels",
            "msa_dms",
            "structures",
            "recommended",
            "all",
        ],
        help='Download ProteinGym data. "recommended" downloads essential datasets for protein_sequence evaluation',
    )
    parser.add_argument(
        "--prepare_assay",
        type=str,
        help="Prepare evaluation data for specific assay ID",
    )

    # バランスサンプリング関連のオプション
    parser.add_argument(
        "--balanced_sampling",
        action="store_true",
        help="Use balanced sampling (extract equal positive and negative samples)",
    )
    parser.add_argument(
        "--positive_samples",
        type=int,
        default=1000,
        help="Number of positive samples to extract (default: 1000)",
    )
    parser.add_argument(
        "--negative_samples",
        type=int,
        default=1000,
        help="Number of negative samples to extract (default: 1000)",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Score threshold for positive/negative classification (default: median)",
    )
    parser.add_argument(
        "--prepare_multiple_assays",
        nargs="+",
        help="Prepare balanced data for multiple assay IDs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for prepared datasets",
    )
    parser.add_argument(
        "--setup_test_data",
        action="store_true",
        help="Setup test data from metadata (create sample balanced datasets)",
    )
    parser.add_argument(
        "--test_assay_count",
        type=int,
        default=5,
        help="Number of test assays to create (default: 5)",
    )
    parser.add_argument("--max_variants", type=int, help="Maximum number of variants to include")
    parser.add_argument("--create_test", type=str, help="Create test dataset (provide output filename)")
    parser.add_argument("--test_size", type=int, default=100, help="Size of test dataset")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    parser.add_argument(
        "--list_assays",
        action="store_true",
        help="List available assays from reference file",
    )
    parser.add_argument(
        "--get_test_assays",
        type=int,
        default=5,
        help="Get small test assays (specify number of assays)",
    )
    parser.add_argument(
        "--data_type",
        choices=["substitutions", "indels"],
        default="substitutions",
        help="Data type for reference file loading",
    )

    args = parser.parse_args()

    # デフォルトのdata_dirを環境変数から設定
    if args.data_dir is None:
        args.data_dir = f"{learning_source_dir}/protein_sequence/gym"

    # data_dirの存在確認
    if not os.path.exists(os.path.dirname(args.data_dir)):
        print(f"❌ ERROR: Parent directory does not exist: {os.path.dirname(args.data_dir)}")
        print(f"Expected structure: {learning_source_dir}/protein_sequence/")
        print("")
        print("Please verify that:")
        print(f"1. LEARNING_SOURCE_DIR='{learning_source_dir}' is correct")
        print("2. The protein_sequence directory structure exists")
        return 1

    downloader = ProteinGymDataDownloader(data_dir=args.data_dir)

    try:
        # ダウンロード
        if args.download:
            if args.download == "recommended":
                # protein_sequence評価に推奨されるデータセットをダウンロード
                downloaded_paths = downloader.download_recommended_datasets(force_download=args.force)
                logger.info("Recommended datasets downloaded:")
                for dataset, path in downloaded_paths.items():
                    if path:
                        logger.info(f"  ✓ {dataset}: {path}")
                    else:
                        logger.warning(f"  ✗ {dataset}: Failed to download")
            elif args.download == "all":
                # 主要なデータセットをすべてダウンロード
                main_datasets = [
                    "substitutions",
                    "indels",
                    "reference_substitutions",
                    "reference_indels",
                    "clinical_substitutions",
                    "clinical_indels",
                ]
                for data_type in main_datasets:
                    try:
                        downloader.download_proteingym_data(data_type, force_download=args.force)
                    except Exception as e:
                        logger.warning(f"Failed to download {data_type}: {e}")
            else:
                downloader.download_proteingym_data(args.download, force_download=args.force)

        # アッセイリスト表示
        if args.list_assays:
            ref_df = downloader.load_reference_file(data_type=args.data_type)
            print(f"\nAvailable {args.data_type} assays:")
            for _, row in ref_df.iterrows():
                protein_id = row.get("UniProt_ID", row.get("protein_name", "N/A"))
                n_variants = row.get("DMS_total_number_mutants", "N/A")
                organism = row.get("taxon", "N/A")
                print(f"  {row['DMS_id']}: {protein_id} ({organism}) - {n_variants} variants")

            print(f"\nTotal assays: {len(ref_df)}")

        # テスト用アッセイの取得
        if args.get_test_assays:
            ref_df = downloader.load_reference_file(data_type=args.data_type)
            test_assays = downloader.get_small_test_assays(ref_df, max_assays=args.get_test_assays)

            print("\nRecommended test assays for quick evaluation:")
            for assay_id in test_assays:
                print(f"  {assay_id}")

            # サンプル実行コマンドを表示
            if test_assays:
                print("\nExample evaluation command:")
                print("python scripts/proteingym_evaluation.py \\")
                print("    --model_path gpt2-output/protein_sequence-small/ckpt.pt \\")
                print(f"    --proteingym_data proteingym_data/{test_assays[0]}.csv \\")
                print(f"    --output_dir results_{test_assays[0]}/")
                print("    --batch_size 16")

        # アッセイデータ準備（単一アッセイ）
        if args.prepare_assay:
            eval_data = downloader.prepare_evaluation_data(
                assay_id=args.prepare_assay,
                max_variants=args.max_variants,
                balanced_sampling=args.balanced_sampling,
                positive_samples=args.positive_samples,
                negative_samples=args.negative_samples,
                score_threshold=args.score_threshold,
            )

            if args.balanced_sampling:
                output_file = f"{args.prepare_assay}_balanced_evaluation_data.csv"
            else:
                output_file = f"{args.prepare_assay}_evaluation_data.csv"

            eval_data.to_csv(output_file, index=False)
            logger.info(f"Evaluation data saved to: {output_file}")

            # 統計情報を表示
            if args.balanced_sampling and args.score_threshold:
                threshold = args.score_threshold
            else:
                threshold = eval_data["DMS_score"].median()

            positive_count = len(eval_data[eval_data["DMS_score"] >= threshold])
            negative_count = len(eval_data[eval_data["DMS_score"] < threshold])
            logger.info("Final dataset statistics:")
            logger.info(f"  Total samples: {len(eval_data)}")
            logger.info(f"  Positive samples (>= {threshold:.3f}): {positive_count}")
            logger.info(f"  Negative samples (< {threshold:.3f}): {negative_count}")

        # 複数アッセイの一括バランス抽出
        if args.prepare_multiple_assays:
            balanced_datasets = downloader.prepare_multiple_assays_balanced(
                assay_ids=args.prepare_multiple_assays,
                positive_samples=args.positive_samples,
                negative_samples=args.negative_samples,
                score_threshold=args.score_threshold,
                output_dir=args.output_dir or "./balanced_proteingym_data",
            )

            logger.info(f"Prepared balanced datasets for {len(balanced_datasets)} assays:")
            for assay_id, df in balanced_datasets.items():
                logger.info(f"  {assay_id}: {len(df)} variants")

        # テストデータセットの作成（メタデータから）
        if args.setup_test_data:
            try:
                created_datasets = downloader.setup_test_data_from_metadata(
                    positive_samples=args.positive_samples,
                    negative_samples=args.negative_samples,
                    test_assay_count=args.test_assay_count,
                )

                logger.info(f"Successfully created {len(created_datasets)} test datasets:")
                for assay_id, info in created_datasets.items():
                    logger.info(
                        f"  {assay_id}: {info['total_samples']} samples "
                        + f"({info['positive_samples']} positive, {info['negative_samples']} negative)"
                    )
                    logger.info(f"    File: {info['file']}")

                # 使用例を表示
                if created_datasets:
                    first_assay = list(created_datasets.keys())[0]
                    first_file = created_datasets[first_assay]["file"]
                    logger.info("\nExample usage:")
                    logger.info("python scripts/proteingym_evaluation.py \\")
                    logger.info("  --model_path runs_train_gpt2_protein_sequence/checkpoint-5000 \\")
                    logger.info(f"  --proteingym_data {first_file}")

            except Exception as e:
                logger.error(f"Failed to create test data: {e}")
                return 1

        # テストデータセット作成
        if args.create_test:
            downloader.create_test_dataset(args.create_test, n_variants=args.test_size)

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
