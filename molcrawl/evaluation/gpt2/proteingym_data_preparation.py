#!/usr/bin/env python3
"""
ProteinGym dataset download and preprocessing utility

This script downloads the ProteinGym dataset and
Preprocess it into a format suitable for evaluation.
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

# Add common module
from molcrawl.core.utils.environment_check import check_learning_source_dir

# Log settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProteinGymDataDownloader:
    """ProteinGym dataset download and preprocessing class"""

    # Official URL of ProteinGym v1.3 dataset
    PROTEINGYM_URLS = {
        # DMS (Deep Mutational Scanning) data - for main evaluation
        "substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip",
        "indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_indels.zip",
        # Reference file - assay metadata
        "reference_substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_substitutions.csv",
        "reference_indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_indels.csv",
        # Clinical variation data - for complementary evaluation
        "clinical_substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_ProteinGym_substitutions.zip",
        "clinical_indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_ProteinGym_indels.zip",
        "clinical_reference_substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_substitutions.csv",
        "clinical_reference_indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_indels.csv",
        # Raw data (if necessary)
        "raw_substitutions": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/substitutions_raw_DMS.zip",
        "raw_indels": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/indels_raw_DMS.zip",
        # Multiple sequence alignment (for advanced analysis)
        "msa_dms": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip",
        "msa_clinical": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/clinical_msa_files.zip",
        # Protein structure (for structure-based analysis)
        "structures": "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/ProteinGym_AF2_structures.zip",
    }

    def __init__(self, data_dir="./proteingym_data"):
        """
        initialization

        Args:
            data_dir (str): Data storage directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filename=None, force_download=False):
        """
        Download file

        Args:
            url (str): Download URL
            filename (str): Save file name（NoneIn the case ofURLestimated from)
            force_download (bool): Overwrite existing file?

        Returns:
            str: path of downloaded file
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
        Extract the ZIP file

        Args:
            zip_path (str): ZIP file path
            extract_dir (str): Extraction destination directory (same directory if None)

        Returns:
            str: extraction destination directory
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
        Download the ProteinGym dataset

        Args:
            data_type (str): data type
                          - 'substitutions': DMS single substitution data (recommended)
                          - 'indels': DMS insertion/deletion data
                          - 'clinical_substitutions': clinical variant substitution data
                          - 'clinical_indels': clinical mutation insertion/deletion data
                          - 'reference_substitutions': DMS substitution reference file
                          - 'reference_indels': DMS insertion/deletion reference file
                          - 'msa_dms': Multiple Sequence Alignment (DMS)
                          - 'structures': protein structure data
            force_download (bool): Force download

        Returns:
            str: path of downloaded file/directory
        """
        if data_type not in self.PROTEINGYM_URLS:
            available_types = list(self.PROTEINGYM_URLS.keys())
            raise ValueError(f"Invalid data_type: {data_type}. Choose from {available_types}")

        url = self.PROTEINGYM_URLS[data_type]
        downloaded_file = self.download_file(url, force_download=force_download)

        # If it is a ZIP file, extract it
        if downloaded_file.endswith(".zip"):
            extracted_dir = self.extract_zip(downloaded_file)
            return extracted_dir
        else:
            return downloaded_file

    def load_reference_file(self, reference_path=None, data_type="substitutions"):
        """
        Load ProteinGym reference file

        Args:
            reference_path (str): Reference file path (automatically downloaded if None)
            data_type (str): data type ('substitutions' or 'indels')

        Returns:
            pd.DataFrame: Reference data
        """
        if reference_path is None:
            reference_key = f"reference_{data_type}"
            if reference_key not in self.PROTEINGYM_URLS:
                reference_key = "reference_substitutions"  # default
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
        Prepare evaluation data for specific assays

        Args:
            assay_id (str): Assay ID
            data_dir (str): data directory（None(automatic download if
            max_variants (int): Maximum number of variants (None for no limit)
            balanced_sampling (bool): Use balanced sampling?
            positive_samples (int): Number of positive samples（balanced_sampling=Truein the case of)
            negative_samples (int): Number of negative samples（balanced_sampling=Truein the case of)
            score_threshold (float): Positive/negative threshold（None(use median value)

        Returns:
            pd.DataFrame: Evaluation data
        """
        if data_dir is None:
            data_dir = self.download_proteingym_data("substitutions")

        # Find assay file
        assay_file = None
        data_path = Path(data_dir)

        for file_path in data_path.rglob(f"{assay_id}.csv"):
            assay_file = file_path
            break

        if assay_file is None:
            raise FileNotFoundError(f"Assay file not found for ID: {assay_id}")

        logger.info(f"Loading assay data: {assay_file}")
        df = pd.read_csv(assay_file)

        # Check required columns
        required_columns = ["mutant", "mutated_sequence", "DMS_score"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # data cleaning
        df = df.dropna(subset=["DMS_score"])
        original_size = len(df)

        # Execute balanced sampling
        if balanced_sampling:
            df = self._balanced_sampling(df, positive_samples, negative_samples, score_threshold)
            logger.info(f"Applied balanced sampling: {original_size} → {len(df)} variants")
        elif max_variants and len(df) > max_variants:
            # Traditional random sampling
            logger.info(f"Limiting to {max_variants} variants (from {len(df)})")
            df = df.sample(n=max_variants, random_state=42)

        logger.info(f"Prepared {len(df)} variants for evaluation")
        logger.info(f"DMS score range: {df['DMS_score'].min():.3f} to {df['DMS_score'].max():.3f}")

        return df

    def create_test_dataset(self, output_file, n_variants=100):
        """
        Create a small dataset for testing

        Args:
            output_file (str): Output file path
            n_variants (int): number of variants
        """
        logger.info(f"Creating test dataset with {n_variants} variants")

        # Sample protein sequence
        base_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAAFRVTLNEKLATWTEESS"

        # generate random mutations
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        data = []

        np.random.seed(42)

        for i in range(n_variants):
            if i == 0:
                # wild type
                mutant = "WT"
                mutated_seq = base_sequence
                score = 1.0
            else:
                # random mutation
                pos = np.random.randint(1, len(base_sequence) + 1)
                orig_aa = base_sequence[pos - 1]
                mut_aa = np.random.choice([aa for aa in amino_acids if aa != orig_aa])

                mutant = f"{orig_aa}{pos}{mut_aa}"
                mutated_seq = base_sequence[: pos - 1] + mut_aa + base_sequence[pos:]

                # Random scores (more realistic distribution)
                score = np.random.beta(2, 5)  # distribution biased towards 0

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
        Extract positive and negative samples in a well-balanced manner

        Args:
            df (pd.DataFrame): Original data frame
            positive_samples (int): Number of positive samples
            negative_samples (int): Number of negative samples
            score_threshold (float): Positive/negative threshold（None(use median value)

        Returns:
            pd.DataFrame: Balanced extracted data frame
        """
        if score_threshold is None:
            score_threshold = df["DMS_score"].median()
            logger.info(f"Using median as threshold: {score_threshold:.3f}")
        else:
            logger.info(f"Using specified threshold: {score_threshold:.3f}")

        # Label as positive/negative
        positive_df = df[df["DMS_score"] >= score_threshold].copy()
        negative_df = df[df["DMS_score"] < score_threshold].copy()

        logger.info(f"Original distribution: {len(positive_df)} positive, {len(negative_df)} negative")

        # Random sampling from each class
        sampled_dfs = []

        # Extracting positive samples
        if len(positive_df) >= positive_samples:
            positive_sampled = positive_df.sample(n=positive_samples, random_state=42)
            logger.info(f"Sampled {positive_samples} positive samples from {len(positive_df)} available")
        else:
            positive_sampled = positive_df.copy()
            logger.warning(f"Only {len(positive_df)} positive samples available (requested {positive_samples})")

        sampled_dfs.append(positive_sampled)

        # Extraction of negative samples
        if len(negative_df) >= negative_samples:
            negative_sampled = negative_df.sample(n=negative_samples, random_state=42)
            logger.info(f"Sampled {negative_samples} negative samples from {len(negative_df)} available")
        else:
            negative_sampled = negative_df.copy()
            logger.warning(f"Only {len(negative_df)} negative samples available (requested {negative_samples})")

        sampled_dfs.append(negative_sampled)

        # combine and shuffle
        balanced_df = pd.concat(sampled_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Output balance information to log
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
        Prepare balanced data extraction from multiple assays

        Args:
            assay_ids (list): list of assay IDs
            data_dir (str): data directory
            positive_samples (int): Number of positive samples per assay
            negative_samples (int): Number of negative samples per assay
            score_threshold (float): Positive/negative threshold
            output_dir (str): Output directory

        Returns:
            dict: Dictionary of {assay_id: dataframe}
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

                # save to separate file
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
        Create a balanced dataset for testing from metadata files

        Args:
            metadata_file (str): Metadata file path
            positive_samples (int): Number of positive samples
            negative_samples (int): Number of negative samples
            test_assay_count (int): Number of test assays to create

        Returns:
            dict: Information about the created test dataset
        """
        if metadata_file is None:
            # defaultFind the metadata file for
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

        # load metadata
        metadata_df = pd.read_csv(metadata_file)
        logger.info(f"Found {len(metadata_df)} assays in metadata")

        # create test directory
        test_data_dir = Path(self.data_dir) / "balanced_evaluation_data"
        test_data_dir.mkdir(exist_ok=True)

        # Select one for testing from the top assays
        test_assays = metadata_df.head(test_assay_count)
        created_datasets = {}

        for _, row in test_assays.iterrows():
            assay_id = row["DMS_id"]
            target_sequence = row.get("target_seq", "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL")  # Default array

            logger.info(f"Creating test data for assay: {assay_id}")

            # Generate sample mutation data
            test_data = self._generate_sample_mutations(
                assay_id=assay_id,
                target_seq=target_sequence,
                positive_samples=positive_samples,
                negative_samples=negative_samples,
            )

            # save to file
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
        Generate sample mutation data for specific assays

        Args:
            assay_id (str): Assay ID
            target_seq (str): target sequence
            positive_samples (int): Number of positive samples
            negative_samples (int): Number of negative samples

        Returns:
            pd.DataFrame: Generated mutation data
        """
        import random

        mutations = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # Set random seed (for reproducibility)
        random.seed(42)
        np.random.seed(42)

        # Positive sample generation (high DMS_score: 0.5~1.5)
        for _i in range(positive_samples):
            pos = random.randint(1, min(len(target_seq), 200))  # Array length limit
            if pos <= len(target_seq):
                orig_aa = target_seq[pos - 1]
                mut_aa = random.choice([aa for aa in amino_acids if aa != orig_aa])

                # create mutant array
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
                    "DMS_score": np.random.uniform(0.5, 1.5),  # Positive score
                    "protein_name": assay_id,
                    "DMS_id": assay_id,
                }
            )

        # Negative sample generation (low DMS_score: -1.5~0.0)
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
                    "DMS_score": np.random.uniform(-1.5, 0.0),  # Negative score
                    "protein_name": assay_id,
                    "DMS_id": assay_id,
                }
            )

        # Create a data frame and shuffle it
        df = pd.DataFrame(mutations)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df

    def download_recommended_datasets(self, force_download=False):
        """
        Download recommended datasets for protein_sequence evaluation

        Args:
            force_download (bool): Force download

        Returns:
            dict: dictionary of downloaded file paths
        """
        logger.info("Downloading recommended datasets for protein_sequence evaluation...")

        recommended_datasets = [
            "substitutions",  # For main evaluation: DMS single substitution data
            "reference_substitutions",  # assay metadata
            "clinical_substitutions",  # For complementary evaluation: clinical mutation data
            "clinical_reference_substitutions",  # Clinical variant metadata
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
        Select a small assay for testing

        Args:
            reference_df (pd.DataFrame): Reference data frame
            max_assays (int): Maximum number of assays
            max_variants_per_assay (int): Maximum number of variants per assay

        Returns:
            list: list of selected assay IDs
        """
        # Filter by number of mutations
        if "DMS_total_number_mutants" in reference_df.columns:
            filtered_df = reference_df[
                (reference_df["DMS_total_number_mutants"] <= max_variants_per_assay)
                & (reference_df["DMS_total_number_mutants"] >= 50)  # Minimum 50 mutations
            ]
        else:
            filtered_df = reference_df

        # randomly selected
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
    # Setting LEARNING_SOURCE_DIR
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

    # Options related to balanced sampling
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

    # Set default data_dir from environment variable
    if args.data_dir is None:
        args.data_dir = f"{learning_source_dir}/protein_sequence/gym"

    # Check the existence of data_dir
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
        if args.download:
            if args.download == "recommended":
                # Download the recommended dataset for protein_sequence evaluation
                downloaded_paths = downloader.download_recommended_datasets(force_download=args.force)
                logger.info("Recommended datasets downloaded:")
                for dataset, path in downloaded_paths.items():
                    if path:
                        logger.info(f"  ✓ {dataset}: {path}")
                    else:
                        logger.warning(f"  ✗ {dataset}: Failed to download")
            elif args.download == "all":
                # Download all major datasets
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

        # Display assay list
        if args.list_assays:
            ref_df = downloader.load_reference_file(data_type=args.data_type)
            print(f"\nAvailable {args.data_type} assays:")
            for _, row in ref_df.iterrows():
                protein_id = row.get("UniProt_ID", row.get("protein_name", "N/A"))
                n_variants = row.get("DMS_total_number_mutants", "N/A")
                organism = row.get("taxon", "N/A")
                print(f"  {row['DMS_id']}: {protein_id} ({organism}) - {n_variants} variants")

            print(f"\nTotal assays: {len(ref_df)}")

        # Get assay for testing
        if args.get_test_assays:
            ref_df = downloader.load_reference_file(data_type=args.data_type)
            test_assays = downloader.get_small_test_assays(ref_df, max_assays=args.get_test_assays)

            print("\nRecommended test assays for quick evaluation:")
            for assay_id in test_assays:
                print(f"  {assay_id}")

            # Display sample execution command
            if test_assays:
                print("\nExample evaluation command:")
                print("python scripts/proteingym_evaluation.py \\")
                print("    --model_path gpt2-output/protein_sequence-small/ckpt.pt \\")
                print(f"    --proteingym_data proteingym_data/{test_assays[0]}.csv \\")
                print(f"    --output_dir results_{test_assays[0]}/")
                print("    --batch_size 16")

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

            # show statistics
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

        # Batch balance extraction of multiple assays
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

        # Create test dataset (from metadata)
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

                # Show usage example
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

        # Create test data set
        if args.create_test:
            downloader.create_test_dataset(args.create_test, n_variants=args.test_size)

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
