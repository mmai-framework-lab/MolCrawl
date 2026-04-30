#!/usr/bin/env python3
"""
COSMIC data download/preprocessing script

Download cancer-related mutation data from the COSMIC database,
Preprocess the genome sequence model into a format suitable for evaluation.

Notice: LEARNING_SOURCE_DIRSetting environment variables is required.
"""

import argparse
import gzip
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests


from molcrawl.core.utils.environment_check import check_learning_source_dir

# Log settings
learning_source_dir = check_learning_source_dir()
log_dir = os.path.join(learning_source_dir, "genome_sequence", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/cosmic_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"LEARNING_SOURCE_DIR: {learning_source_dir}")


class COSMICProcessor:
    """COSMIC data acquisition/preprocessing class"""

    def __init__(self, output_dir=None):
        """
        initialization

        Args:
            output_dir (str): Output directory（NoneIn the case of$LEARNING_SOURCE_DIR/genome_sequence/data/cosmic）
        """
        if output_dir is None:
            learning_source = check_learning_source_dir()
            output_dir = os.path.join(learning_source, "genome_sequence", "data", "cosmic")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

        # COSMIC public data URL (dataset that does not require authentication)
        self.cosmic_urls = {
            "census": "https://cancer.sanger.ac.uk/cosmic/file_download/GRCh38/cosmic/v97/Cancer_Gene_Census.csv",
            "mutations": "https://cancer.sanger.ac.uk/cosmic/file_download/GRCh38/cosmic/v97/CosmicMutantExport.tsv.gz",
        }

        # Cancer severity classification
        self.cancer_significance_map = {
            "pathogenic": ["pathogenic", "likely_pathogenic", "oncogenic"],
            "benign": ["benign", "likely_benign", "neutral"],
            "uncertain": ["uncertain_significance", "conflicting", "unknown"],
        }

    def download_cosmic_data(self, dataset_type="census"):
        """
        Download COSMIC data

        Args:
            dataset_type (str): dataset type ('census', 'mutations')
        """
        if dataset_type not in self.cosmic_urls:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        url = self.cosmic_urls[dataset_type]
        output_file = self.output_dir / f"cosmic_{dataset_type}.{'csv' if dataset_type == 'census' else 'tsv.gz'}"

        logger.info(f"Downloading COSMIC {dataset_type} data from {url}")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to download {dataset_type}: {e}")
            return None

    def create_sample_cosmic_data(self, num_samples=20):
        """
        Create sample COSMIC data (for testing)

        Args:
            num_samples (int): Number of samples to create
        """
        logger.info(f"Creating sample COSMIC data with {num_samples} samples")

        # Sample data based on actual COSMIC mutation patterns
        sample_data = []

        # Known cancer-related genes and mutation patterns
        cancer_genes = [
            "TP53",
            "KRAS",
            "PIK3CA",
            "APC",
            "BRCA1",
            "BRCA2",
            "EGFR",
            "MYC",
        ]
        mutation_types = [
            "Substitution - Missense",
            "Substitution - Nonsense",
            "Deletion - Frameshift",
            "Insertion - Frameshift",
        ]

        for i in range(num_samples):
            gene = np.random.choice(cancer_genes)
            mutation_type = np.random.choice(mutation_types)

            # Generate DNA sequence (100bp)
            bases = ["A", "T", "G", "C"]
            ref_sequence = "".join(np.random.choice(bases, 100))

            # Introducing mutations
            var_sequence = list(ref_sequence)
            mutation_pos = np.random.randint(45, 55)  # Mutation near the center

            if "Missense" in mutation_type or "Nonsense" in mutation_type:
                # point mutation
                original_base = var_sequence[mutation_pos]
                possible_bases = [b for b in bases if b != original_base]
                var_sequence[mutation_pos] = np.random.choice(possible_bases)
            elif "Deletion" in mutation_type:
                # deletion
                del_length = np.random.randint(1, 4)
                del var_sequence[mutation_pos : mutation_pos + del_length]
            elif "Insertion" in mutation_type:
                # insert
                ins_length = np.random.randint(1, 4)
                insert_bases = "".join(np.random.choice(bases, ins_length))
                var_sequence.insert(mutation_pos, insert_bases)

            var_sequence = "".join(var_sequence)

            # Determination of cancer relevance (based on gene and mutation type)
            if gene in ["TP53", "BRCA1", "BRCA2"] and "Nonsense" in mutation_type:
                significance = "pathogenic"
                oncogenic = 1
            elif gene in ["KRAS", "PIK3CA"] and "Missense" in mutation_type:
                significance = "likely_pathogenic" if np.random.random() > 0.3 else "pathogenic"
                oncogenic = 1
            elif "Frameshift" in mutation_type:
                significance = "pathogenic" if np.random.random() > 0.2 else "likely_pathogenic"
                oncogenic = 1
            else:
                significance = np.random.choice(["benign", "likely_benign", "uncertain_significance"])
                oncogenic = 0

            sample_data.append(
                {
                    "COSMIC_ID": f"COSV{1000000 + i}",
                    "Gene_name": gene,
                    "Mutation_Type": mutation_type,
                    "Cancer_significance": significance,
                    "oncogenic": oncogenic,
                    "Chromosome": f"chr{np.random.randint(1, 23)}",
                    "Position": np.random.randint(1000000, 200000000),
                    "Reference_sequence": ref_sequence,
                    "Variant_sequence": var_sequence,
                    "Primary_site": np.random.choice(["lung", "breast", "colon", "prostate", "liver"]),
                    "Sample_count": np.random.randint(1, 50),
                    "Mutation_somatic_status": "Confirmed somatic",
                }
            )

        # convert to DataFrame
        df = pd.DataFrame(sample_data)

        # Save as CSV file
        output_file = self.output_dir / "cosmic_evaluation_dataset.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"Sample COSMIC data saved to {output_file}")
        logger.info(f"Data distribution: {df['oncogenic'].value_counts().to_dict()}")

        return output_file

    def parse_cosmic_vcf(self, vcf_file):
        """
        Analyze COSMIC VCF files

        Args:
            vcf_file (str): VCF file path

        Returns:
            pd.DataFrame: parsed data
        """
        logger.info(f"Parsing COSMIC VCF file: {vcf_file}")

        mutations = []

        try:
            opener = gzip.open if str(vcf_file).endswith(".gz") else open
            with opener(vcf_file, "rt") as f:
                for line_num, line in enumerate(f):
                    if line.startswith("#"):
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) < 8:
                        continue

                    chrom = parts[0]
                    pos = int(parts[1])
                    ref = parts[3]
                    alt = parts[4]
                    info = parts[7]

                    # extract information from INFO field
                    info_dict = {}
                    for item in info.split(";"):
                        if "=" in item:
                            key, value = item.split("=", 1)
                            info_dict[key] = value

                    mutations.append(
                        {
                            "chromosome": chrom,
                            "position": pos,
                            "reference_allele": ref,
                            "alternate_allele": alt,
                            "gene": info_dict.get("GENE", ""),
                            "cosmic_id": info_dict.get("COSMIC_ID", ""),
                            "mutation_type": info_dict.get("VARIANT_CLASS", ""),
                            "cancer_type": info_dict.get("CANCER_TYPE", ""),
                            "sample_count": int(info_dict.get("CNT", 1)),
                        }
                    )

                    # Memory usage limit
                    if line_num > 100000:  # Process only the first 100,000 lines
                        logger.info("Limiting to first 100k variants for memory efficiency")
                        break

        except Exception as e:
            logger.error(f"Error parsing VCF file: {e}")
            return pd.DataFrame()

        return pd.DataFrame(mutations)

    def generate_sequences_from_variants(self, df, sequence_length=100):
        """
        Generate reference and mutant sequences from mutation information

        Args:
            df (pd.DataFrame): variant data
            sequence_length (int): length of the sequence to generate

        Returns:
            pd.DataFrame: Data with array information added
        """
        logger.info(f"Generating sequences of length {sequence_length}")

        sequences = []
        bases = ["A", "T", "G", "C"]

        for _, row in df.iterrows():
            # Generate random context array
            half_len = sequence_length // 2

            # Generate reference array
            prefix = "".join(np.random.choice(bases, half_len - len(row["reference_allele"]) // 2))
            suffix = "".join(np.random.choice(bases, half_len - len(row["reference_allele"]) // 2))
            ref_sequence = prefix + row["reference_allele"] + suffix

            # Generate mutant array
            var_sequence = prefix + row["alternate_allele"] + suffix

            # Adjust length
            if len(ref_sequence) < sequence_length:
                ref_sequence += "".join(np.random.choice(bases, sequence_length - len(ref_sequence)))
            elif len(ref_sequence) > sequence_length:
                ref_sequence = ref_sequence[:sequence_length]

            if len(var_sequence) < sequence_length:
                var_sequence += "".join(np.random.choice(bases, sequence_length - len(var_sequence)))
            elif len(var_sequence) > sequence_length:
                var_sequence = var_sequence[:sequence_length]

            sequences.append({"reference_sequence": ref_sequence, "variant_sequence": var_sequence})

        # Add array information to the original DataFrame
        sequence_df = pd.DataFrame(sequences)
        result_df = pd.concat([df.reset_index(drop=True), sequence_df], axis=1)

        return result_df


def main():
    """Main processing"""
    parser = argparse.ArgumentParser(
        description="COSMIC data preparation for genome sequence evaluation",
        epilog="Note: LEARNING_SOURCE_DIR environment variable must be set.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: $LEARNING_SOURCE_DIR/genome_sequence/data/cosmic)",
    )
    parser.add_argument("--download", action="store_true", help="Download COSMIC data")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples per class")
    parser.add_argument("--sequence_length", type=int, default=100, help="Sequence length")
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample data instead of downloading",
    )

    args = parser.parse_args()

    processor = COSMICProcessor(args.output_dir)

    if args.create_sample_data:
        logger.info("Creating sample COSMIC data")
        processor.create_sample_cosmic_data(args.max_samples)
    else:
        logger.info("Sample COSMIC data creation completed")
        logger.info("Note: For real COSMIC data, registration and authentication are required")
        logger.info("Creating sample data for demonstration...")
        processor.create_sample_cosmic_data(args.max_samples)

    logger.info("COSMIC data preparation completed")


if __name__ == "__main__":
    main()
