#!/usr/bin/env python3
"""
ClinVar data download/preprocessing script

Download pathogenic variant data from the ClinVar database,
Preprocess the genome sequence model into a format suitable for evaluation.
"""

import argparse
import gzip
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Wire up the file + stream handlers for command-line use.

    Kept out of the module body so that ``import`` does not have the side
    effect of creating a ``logs/`` directory and a timestamped file —
    importing this module (e.g. from pdoc or a test) used to fail with
    ``FileNotFoundError`` because the relative ``logs/`` path may not
    exist in the caller's cwd.
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/clinvar_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(),
        ],
    )


class ClinVarProcessor:
    """ClinVar data acquisition/preprocessing class"""

    def __init__(self, output_dir="./clinvar_data"):
        """
        initialization

        Args:
            output_dir (str): Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ClinVar download URL
        self.clinvar_urls = {
            "variant_summary": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
            "submission_summary": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/submission_summary.txt.gz",
        }

    def download_clinvar_data(self, data_type="variant_summary"):
        """
        Download ClinVar data

        Args:
            data_type (str): Data type to download

        Returns:
            str: path of the downloaded file
        """
        if data_type not in self.clinvar_urls:
            raise ValueError(f"Unknown data type: {data_type}")

        url = self.clinvar_urls[data_type]
        filename = self.output_dir / f"{data_type}.txt.gz"

        logger.info(f"Downloading {data_type} from {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded to {filename}")
        return str(filename)

    def extract_and_load_data(self, gzip_file):
        """
        Unzip the compressed file and read it as a DataFrame

        Args:
            gzip_file (str): gzip file path

        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Extracting and loading {gzip_file}")

        with gzip.open(gzip_file, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f, sep="\t", low_memory=False)

        logger.info(f"Loaded {len(df)} records")
        return df

    def filter_single_nucleotide_variants(self, df):
        """
        Filter only single nucleotide variations (SNVs)

        Args:
            df (pd.DataFrame): ClinVar data

        Returns:
            pd.DataFrame: Filtered data
        """
        logger.info(f"Starting with {len(df)} total variants")

        # Check available mutation types
        logger.info(f"Available variant types: {df['Type'].value_counts().head(10)}")

        # Filter by mutation type
        snv_types = ["single nucleotide variant", "SNV"]
        df_snv = df[df["Type"].isin(snv_types)].copy()
        logger.info(f"After type filtering: {len(df_snv)} variants")

        if len(df_snv) > 0:
            logger.info(f"Available columns: {df_snv.columns.tolist()}")

            # Use ReferenceAlleleVCF and AlternateAlleleVCF fields
            if "ReferenceAlleleVCF" in df_snv.columns and "AlternateAlleleVCF" in df_snv.columns:
                logger.info("Using VCF allele fields (ReferenceAlleleVCF, AlternateAlleleVCF)")
                # copy VCF field to standard field
                df_snv["ReferenceAllele"] = df_snv["ReferenceAlleleVCF"]
                df_snv["AlternateAllele"] = df_snv["AlternateAlleleVCF"]

            logger.info(f"First few ReferenceAllele values: {df_snv['ReferenceAllele'].head(10).tolist()}")
            logger.info(f"First few AlternateAllele values: {df_snv['AlternateAllele'].head(10).tolist()}")

        # Filter only valid allele information by excluding "na" value
        valid_alleles = (
            (df_snv["ReferenceAllele"].notna())
            & (df_snv["AlternateAllele"].notna())
            & (df_snv["ReferenceAllele"].astype(str).str.len() > 0)
            & (df_snv["AlternateAllele"].astype(str).str.len() > 0)
            & (df_snv["ReferenceAllele"].astype(str).str.lower() != "na")
            & (df_snv["AlternateAllele"].astype(str).str.lower() != "na")
        )

        df_snv = df_snv[valid_alleles].copy()
        logger.info(f"After removing null/na alleles: {len(df_snv)} variants")

        # Filter only single bases
        if len(df_snv) > 0:
            # Convert to uppercase and process
            df_snv["ReferenceAllele"] = df_snv["ReferenceAllele"].astype(str).str.upper()
            df_snv["AlternateAllele"] = df_snv["AlternateAllele"].astype(str).str.upper()

            valid_ref_len = df_snv["ReferenceAllele"].str.len() == 1
            valid_alt_len = df_snv["AlternateAllele"].str.len() == 1
            valid_ref_base = df_snv["ReferenceAllele"].isin(["A", "T", "G", "C"])
            valid_alt_base = df_snv["AlternateAllele"].isin(["A", "T", "G", "C"])

            logger.info(f"Valid reference length: {valid_ref_len.sum()}")
            logger.info(f"Valid alternate length: {valid_alt_len.sum()}")
            logger.info(f"Valid reference bases: {valid_ref_base.sum()}")
            logger.info(f"Valid alternate bases: {valid_alt_base.sum()}")

            df_snv = df_snv[valid_ref_len & valid_alt_len & valid_ref_base & valid_alt_base].copy()
            logger.info(f"After allele filtering: {len(df_snv)} variants")

            # Check the sample value after filtering
            if len(df_snv) > 0:
                logger.info(f"Final dataset: {len(df_snv)} variants")
                logger.info(f"Final ReferenceAllele sample: {df_snv['ReferenceAllele'].value_counts().head()}")
                logger.info(f"Final AlternateAllele sample: {df_snv['AlternateAllele'].value_counts().head()}")

        return df_snv

    def filter_by_clinical_significance(self, df):
        """
        Filter by clinical significance

        Args:
            df (pd.DataFrame): ClinVar data

        Returns:
            pd.DataFrame: Filtered data
        """
        logger.info(f"Starting clinical significance filtering with {len(df)} variants")

        # Check the actual ClinicalSignificance value
        logger.info("Available clinical significance values:")
        for value, count in df["ClinicalSignificance"].value_counts().head(20).items():
            logger.info(f"  {value}: {count}")

        clear_classifications = [
            "Pathogenic",
            "Likely pathogenic",
            "Benign",
            "Likely benign",
        ]

        df_filtered = df[df["ClinicalSignificance"].isin(clear_classifications)].copy()
        logger.info(f"After clinical significance filtering: {len(df_filtered)} variants")

        return df_filtered

    def get_reference_sequences(self, df, sequence_length=50):
        """
        Get reference array (simplified version - actual implementation uses external API)

        Args:
            df (pd.DataFrame): ClinVar data
            sequence_length (int): array length to obtain

        Returns:
            pd.DataFrame: Data with reference array added
        """
        logger.info("Generating reference sequences (mock implementation)")

        # In actual implementation, use Ensembl REST API or NCBI API
        # you need to get the actual reference array
        # Here we will generate a random array as a sample implementation

        import random

        def generate_mock_sequence(length):
            """Generate DNA sequence for mock"""
            bases = ["A", "T", "G", "C"]
            return "".join(random.choices(bases, k=length))

        def create_variant_sequence(ref_seq, position, ref_allele, alt_allele):
            """Create mutant array"""
            # Place mutation near the center of the array
            mid_pos = len(ref_seq) // 2
            variant_seq = ref_seq[:mid_pos] + alt_allele + ref_seq[mid_pos + 1 :]
            return variant_seq

        reference_sequences = []
        variant_sequences = []

        for _, row in df.iterrows():
            # Generate mock reference array
            ref_seq = generate_mock_sequence(sequence_length)

            # create mutant array
            var_seq = create_variant_sequence(
                ref_seq,
                sequence_length // 2,  # central position
                row["ReferenceAllele"],
                row["AlternateAllele"],
            )

            reference_sequences.append(ref_seq)
            variant_sequences.append(var_seq)

        df_with_sequences = df.copy()
        df_with_sequences["reference_sequence"] = reference_sequences
        df_with_sequences["variant_sequence"] = variant_sequences

        logger.info("Reference sequences generated (mock)")
        return df_with_sequences

    def prepare_evaluation_dataset(self, df, max_samples_per_class=1000):
        """
        Prepare dataset for evaluation

        Args:
            df (pd.DataFrame): Preprocessed ClinVar data
            max_samples_per_class (int): Maximum number of samples per class

        Returns:
            pd.DataFrame: Evaluation dataset
        """
        logger.info("Preparing evaluation dataset")

        # Convert to binary classification of pathogenic/non-pathogenic
        pathogenic_terms = ["Pathogenic", "Likely pathogenic"]
        benign_terms = ["Benign", "Likely benign"]

        df_pathogenic = df[df["ClinicalSignificance"].isin(pathogenic_terms)].copy()
        df_benign = df[df["ClinicalSignificance"].isin(benign_terms)].copy()

        # Number of sampleslimit
        if len(df_pathogenic) > max_samples_per_class:
            df_pathogenic = df_pathogenic.sample(n=max_samples_per_class, random_state=42)

        if len(df_benign) > max_samples_per_class:
            df_benign = df_benign.sample(n=max_samples_per_class, random_state=42)

        # join
        df_eval = pd.concat([df_pathogenic, df_benign], ignore_index=True)

        # add standardized labels
        df_eval["pathogenic"] = df_eval["ClinicalSignificance"].apply(lambda x: 1 if x in pathogenic_terms else 0)

        # keep only necessary columns
        columns_to_keep = [
            "VariationID",
            "GeneSymbol",
            "ClinicalSignificance",
            "pathogenic",
            "Chromosome",
            "Start",
            "ReferenceAllele",
            "AlternateAllele",
            "reference_sequence",
            "variant_sequence",
        ]

        df_eval = df_eval[columns_to_keep].copy()

        # shuffle data
        df_eval = df_eval.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Prepared evaluation dataset with {len(df_eval)} variants")
        logger.info(f"Pathogenic: {len(df_pathogenic)}, Benign: {len(df_benign)}")

        return df_eval

    def save_dataset(self, df, filename):
        """
        Save dataset

        Args:
            df (pd.DataFrame): Data set to save
            filename (str): file name
        """
        filepath = self.output_dir / filename

        # Save in multiple formats
        df.to_csv(str(filepath).replace(".csv", ".csv"), index=False)
        df.to_json(str(filepath).replace(".csv", ".json"), orient="records", indent=2)

        logger.info(f"Dataset saved to {filepath} (CSV and JSON formats)")

    def create_statistics_report(self, df):
        """
        Create statistical reports for datasets

        Args:
            df (pd.DataFrame): Evaluation dataset
        """
        report_file = self.output_dir / "dataset_statistics.txt"

        with open(report_file, "w") as f:
            f.write("ClinVar Dataset Statistics Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total variants: {len(df)}\n")
            f.write(f"Pathogenic variants: {len(df[df['pathogenic'] == 1])}\n")
            f.write(f"Benign variants: {len(df[df['pathogenic'] == 0])}\n\n")

            f.write("Clinical Significance Distribution:\n")
            for sig, count in df["ClinicalSignificance"].value_counts().items():
                f.write(f"  {sig}: {count}\n")
            f.write("\n")

            f.write("Chromosome Distribution (top 10):\n")
            for chrom, count in df["Chromosome"].value_counts().head(10).items():
                f.write(f"  {chrom}: {count}\n")
            f.write("\n")

            f.write("Gene Distribution (top 10):\n")
            for gene, count in df["GeneSymbol"].value_counts().head(10).items():
                f.write(f"  {gene}: {count}\n")
            f.write("\n")

            f.write("Reference/Alternate Allele Distribution:\n")
            f.write("Reference Alleles:\n")
            for allele, count in df["ReferenceAllele"].value_counts().items():
                f.write(f"  {allele}: {count}\n")
            f.write("Alternate Alleles:\n")
            for allele, count in df["AlternateAllele"].value_counts().items():
                f.write(f"  {allele}: {count}\n")

        logger.info(f"Statistics report saved to {report_file}")


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(description="ClinVar data preprocessing for genome sequence evaluation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./clinvar_data",
        help="Output directory for processed data",
    )
    parser.add_argument("--download", action="store_true", help="Download ClinVar data from NCBI")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples per class")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=100,
        help="Length of reference sequences to generate",
    )
    parser.add_argument("--input_file", type=str, help="Input ClinVar file (if not downloading)")

    args = parser.parse_args()

    # create log directory
    os.makedirs("logs", exist_ok=True)

    processor = ClinVarProcessor(args.output_dir)

    try:
        if args.download:
            logger.info("Downloading ClinVar data")
            gzip_file = processor.download_clinvar_data("variant_summary")
            df = processor.extract_and_load_data(gzip_file)
        elif args.input_file:
            logger.info(f"Loading data from {args.input_file}")
            if args.input_file.endswith(".gz"):
                df = processor.extract_and_load_data(args.input_file)
            else:
                df = pd.read_csv(args.input_file, sep="\t", low_memory=False)
        else:
            raise ValueError("Either --download or --input_file must be specified")

        logger.info(f"Initial dataset size: {len(df)}")

        logger.info("Starting data preprocessing")

        # Filtering for single nucleotide variations
        df_snv = processor.filter_single_nucleotide_variants(df)

        # Filter by clinical significance
        df_filtered = processor.filter_by_clinical_significance(df_snv)

        # Get reference array (mock implementation)
        df_with_sequences = processor.get_reference_sequences(df_filtered, sequence_length=args.sequence_length)

        # Prepare dataset for evaluation
        df_eval = processor.prepare_evaluation_dataset(df_with_sequences, max_samples_per_class=args.max_samples)

        # Save dataset
        processor.save_dataset(df_eval, "clinvar_evaluation_dataset.csv")

        # Create statistics report
        processor.create_statistics_report(df_eval)

        logger.info("Data preprocessing completed successfully")
        logger.info(f"Final dataset saved with {len(df_eval)} variants")

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
