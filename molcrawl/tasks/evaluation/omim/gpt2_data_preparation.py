#!/usr/bin/env python3
"""
OMIM Data Preparation Script
============================

Genetic disease information from the OMIM (Online Mendelian Inheritance in Man) database
Script to preprocess genome sequence models for evaluation

Main features:
- Generation of OMIM sample data
- Processing of actual OMIM data (authenticated access)
- Classification of genetic disease-related mutations and Benin mutations
- Creation of dataset for model evaluation
- Data quality check

Notice:
- License required for actual OMIM data usage
- Setting the LEARNING_SOURCE_DIR environment variable is mandatory
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from molcrawl.tasks.evaluation.omim.gpt2_real_data_processor import process_omim_real_data
from molcrawl.core.utils.environment_check import check_learning_source_dir


def setup_logging(output_dir: str) -> logging.Logger:
    """Set up log settings"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"omim_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    learning_source_dir = check_learning_source_dir()
    logger.info(f"LEARNING_SOURCE_DIR: {learning_source_dir}")
    logger.info(f"Output directory: {output_dir}")
    return logger


class OMIMDataGenerator:
    """Sample data generation class for OMIM evaluation"""

    def __init__(self, sequence_length: int = 100, logger: Optional[logging.Logger] = None):
        self.sequence_length = sequence_length
        self.logger = logger or logging.getLogger(__name__)

        # Examples of OMIM-related genes (representative genes obtained from actual OMIM)
        self.disease_genes = [
            "BRCA1",
            "BRCA2",
            "TP53",
            "APC",
            "MLH1",
            "MSH2",
            "MSH6",
            "PMS2",
            "PALB2",
            "CHEK2",
            "ATM",
            "BARD1",
            "BRIP1",
            "RAD51C",
            "RAD51D",
            "CDH1",
            "PTEN",
            "STK11",
            "NF1",
            "NF2",
            "VHL",
            "RB1",
            "WT1",
            "CDKN2A",
            "CDK4",
            "BAP1",
            "MITF",
            "POT1",
            "ACD",
            "TERF2IP",
            "TERT",
            "DCC",
            "SMAD4",
            "BMPR1A",
            "MUTYH",
            "NTHL1",
            "POLE",
            "POLD1",
            "EPCAM",
            "PMS1",
            "AXIN2",
            "GREM1",
            "SCG5",
            "RNF43",
        ]

        # OMIMPhenotype classification
        self.phenotype_categories = {
            "autosomal_dominant": 0.3,
            "autosomal_recessive": 0.25,
            "x_linked": 0.15,
            "mitochondrial": 0.05,
            "complex": 0.25,
        }

        # Pathogenicity level
        self.pathogenicity_levels = {
            "pathogenic": 1,
            "likely_pathogenic": 1,
            "uncertain_significance": 0,
            "likely_benign": 0,
            "benign": 0,
        }

    def generate_sequence(self, is_pathogenic: bool = True) -> str:
        """Generate genome sequence"""
        nucleotides = ["A", "T", "G", "C"]

        if is_pathogenic:
            # Pathogenic mutations: Contains more mutations
            sequence = "".join(random.choices(nucleotides, k=self.sequence_length))
            # Insert pathogenic mutation pattern at specific position
            mutation_positions = random.sample(range(self.sequence_length), min(5, self.sequence_length // 20))
            sequence_list = list(sequence)
            for pos in mutation_positions:
                # Mimic frameshifts and stop codons
                sequence_list[pos] = random.choice(["T", "A"])  # More pathogenic mutations
            sequence = "".join(sequence_list)
        else:
            # Benin mutation: more conservative sequence
            sequence = "".join(random.choices(nucleotides, weights=[0.3, 0.3, 0.2, 0.2], k=self.sequence_length))

        return sequence

    def generate_omim_entry(self, entry_id: int) -> Dict:
        """Generate a single OMIM entry"""
        is_pathogenic = random.random() < 0.7  # 70% pathogenic

        gene = random.choice(self.disease_genes)
        phenotype_type = random.choices(
            list(self.phenotype_categories.keys()),
            weights=list(self.phenotype_categories.values()),
        )[0]

        if is_pathogenic:
            pathogenicity = random.choices(["pathogenic", "likely_pathogenic"], weights=[0.7, 0.3])[0]
        else:
            pathogenicity = random.choices(
                ["benign", "likely_benign", "uncertain_significance"],
                weights=[0.5, 0.3, 0.2],
            )[0]

        # OMIM ID format (6 digit number)
        omim_id = f"{entry_id + 100000:06d}"

        return {
            "omim_id": omim_id,
            "gene_symbol": gene,
            "sequence": self.generate_sequence(is_pathogenic),
            "phenotype_type": phenotype_type,
            "pathogenicity": pathogenicity,
            "is_disease_causing": self.pathogenicity_levels[pathogenicity],
            "chromosome": random.choice([str(i) for i in range(1, 23)] + ["X", "Y"]),
            "position": random.randint(1000000, 200000000),
            "inheritance_pattern": phenotype_type,
            "clinical_significance": pathogenicity,
            "mim_number": omim_id,
            "disease_name": f"Hereditary {gene.lower()} disorder",
            "molecular_basis": "Point mutation" if random.random() < 0.6 else "Deletion/Duplication",
        }

    def generate_dataset(self, num_samples: int) -> pd.DataFrame:
        """Generate OMIM dataset"""
        self.logger.info(f"Creating sample OMIM data with {num_samples} samples")

        data = []
        for i in range(num_samples):
            entry = self.generate_omim_entry(i)
            data.append(entry)

            if (i + 1) % 100 == 0:
                self.logger.info(f"Generated {i + 1}/{num_samples} OMIM entries")

        df = pd.DataFrame(data)

        # Adjust data balance
        pathogenic_count = df["is_disease_causing"].sum()
        benign_count = len(df) - pathogenic_count

        self.logger.info("Generated dataset statistics:")
        self.logger.info(f"  Disease-causing variants: {pathogenic_count}")
        self.logger.info(f"  Benign variants: {benign_count}")
        self.logger.info(f"  Total samples: {len(df)}")

        return df


def prepare_omim_data(output_dir: str, num_samples: int = 1000, sequence_length: int = 100, seed: int = 42) -> str:
    """Prepare OMIM evaluation data"""

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Log settings
    logger = setup_logging(output_dir)
    logger.info("Starting OMIM data preparation")

    # Seed settings
    random.seed(seed)
    np.random.seed(seed)

    # data generation
    generator = OMIMDataGenerator(sequence_length=sequence_length, logger=logger)
    df = generator.generate_dataset(num_samples)

    # Save data
    output_file = os.path.join(data_dir, "omim_evaluation_dataset.csv")
    df.to_csv(output_file, index=False)

    logger.info(f"Sample OMIM data saved to {output_file}")
    logger.info(f"Data distribution: {df['is_disease_causing'].value_counts().to_dict()}")

    # metadatakeep
    metadata = {
        "total_samples": len(df),
        "disease_causing": int(df["is_disease_causing"].sum()),
        "benign": int((df["is_disease_causing"] == 0).sum()),
        "sequence_length": sequence_length,
        "generation_date": datetime.now().isoformat(),
        "inheritance_patterns": df["inheritance_pattern"].value_counts().to_dict(),
        "pathogenicity_distribution": df["pathogenicity"].value_counts().to_dict(),
    }

    metadata_file = os.path.join(data_dir, "omim_metadata.json")
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("OMIM data preparation completed")
    return output_file


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="OMIM Data Preparation for Genome Sequence Evaluation",
        epilog="Note: LEARNING_SOURCE_DIR environment variable must be set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for prepared data (default: $LEARNING_SOURCE_DIR/genome_sequence/data/omim)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "real"],
        default="sample",
        help="Data mode: sample (generated data) or real (actual OMIM data)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to real data config file (required for real mode)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate for sample mode (default: 1000)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=100,
        help="Length of genome sequences (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--existing_omim_dir",
        type=str,
        default=None,
        help="Existing OMIM data directory to use (skip download)",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download for real mode even if files exist",
    )

    args = parser.parse_args()

    # Default setting for output_dir
    if args.output_dir is None:
        learning_source = check_learning_source_dir()
        args.output_dir = os.path.join(learning_source, "genome_sequence", "data", "omim")

    try:
        if args.mode == "real":
            # real data mode
            if not args.config:
                raise ValueError("Real data mode requires --config parameter")

            print("Processing real OMIM data...")
            output_file = process_omim_real_data(
                config_path=args.config,
                output_dir=args.output_dir,
                existing_omim_dir=args.existing_omim_dir,
                force_download=args.force_download,
            )
            print("Real OMIM data processing completed!")

        else:
            # sample data mode
            print("Generating sample OMIM data...")
            output_file = prepare_omim_data(
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                sequence_length=args.sequence_length,
                seed=args.seed,
            )
            print("Sample OMIM data preparation completed!")

        print(f"Output file: {output_file}")

    except Exception as e:
        print(f"Error during OMIM data preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
