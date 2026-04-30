#!/usr/bin/env python3
"""
Protein Classification Data Preparation

Dataset preparation script for protein variant classification evaluation
Created based on the separation principle of data preparation, evaluation, and visualization
"""

import argparse
import logging
import os
import sys
from importlib import import_module

import numpy as np
import pandas as pd

# Set project root and import common modules

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

check_learning_source_dir = import_module("utils.environment_check").check_learning_source_dir

# Logging settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_default_output_dir():
    """Get default output directory"""
    learning_source_dir = check_learning_source_dir()
    return os.path.join(learning_source_dir, "protein_sequence", "data", "protein_classification")


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Create sample protein variation dataset

    Args:
        output_path: Output destination CSV file path
        num_samples: number of samples to generate

    Returns:
        Generated dataset path
    """
    logger.info(f"Creating sample protein classification dataset with {num_samples} samples")

    # Sample protein sequences (various lengths and types)
    sample_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MGSSHHHHHHSSGLVPRGSHMKELKRLTCCKVQTCLRPPGQRQELAYFFKALPQCCNLCSPLVQNPKNCT",
        "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEL",
        "MADEAAQGAFQPGASGSRSKELKEAEDEAEEAEEAKEAEEEAKEAEEEAKEAEEEAKEAEEEA",
        "MGKEKIFSDDVRAIKEQKMLQIKHTAMAEVFLEQLACKMYSVDANTIKDFDLQHIWWNTVEQCE",
    ]

    data = []
    np.random.seed(42)

    for i in range(num_samples):
        # Randomly select an array
        sequence = np.random.choice(sample_sequences)

        # Randomly select mutation position (avoid start/end)
        variant_pos = np.random.randint(5, len(sequence) - 5)
        ref_aa = sequence[variant_pos]

        # Randomly select alternative amino acids
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        alt_aa = np.random.choice([aa for aa in amino_acids if aa != ref_aa])

        # Rule-based assignment of pathogenicity (for demo purposes)
        # Actual data is obtained from a database such as ClinVar
        pathogenic = 0

        # Simple heuristic: certain amino acid mutations are considered more pathogenic
        if ref_aa in "CGHPWY" and alt_aa not in "ACDEFGHIKLMNPQRSTVWY"[:10]:
            pathogenic = 1
        elif variant_pos < len(sequence) * 0.3:  # Ndistal region
            pathogenic = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            pathogenic = np.random.choice([0, 1], p=[0.8, 0.2])

        data.append(
            {
                "variant_id": f"VAR_{i:03d}",
                "sequence": sequence,
                "variant_pos": variant_pos,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "pathogenic": pathogenic,
                "description": f"{ref_aa}{variant_pos + 1}{alt_aa}",
            }
        )

    # convert to DataFrame
    df = pd.DataFrame(data)

    # create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Sample dataset created: {output_path}")
    logger.info(f"   Total samples: {len(df)}")
    logger.info(f"   Pathogenic: {df['pathogenic'].sum()}")
    logger.info(f"   Benign: {len(df) - df['pathogenic'].sum()}")

    return output_path


def prepare_protein_classification_data(input_csv=None, output_dir=None, num_samples=100, create_sample=False):
    """
    Protein classification data preparation main processing

    Args:
        input_csv: Input CSV file (when using existing data)
        output_dir: Output directory
        num_samples: Number of samples when generating sample data
        create_sample: Whether to generate sample data

    Returns:
        Prepared dataset path
    """
    if output_dir is None:
        output_dir = get_default_output_dir()

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    if create_sample:
        # Generate sample data
        output_path = os.path.join(output_dir, "protein_classification_sample.csv")
        return create_sample_dataset(output_path, num_samples)

    elif input_csv:
        # Process existing data (currently simple copy, add preprocessing in the future)
        logger.info(f"Processing existing data: {input_csv}")
        df = pd.read_csv(input_csv)

        # Basic validation
        required_columns = ["variant_id", "sequence", "pathogenic"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        output_path = os.path.join(output_dir, "protein_classification_processed.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Data processed: {output_path}")
        logger.info(f"   Total samples: {len(df)}")

        return output_path

    else:
        raise ValueError("Either --create_sample or --input_csv must be specified")


def main():
    """Main processing"""
    parser = argparse.ArgumentParser(
        description="Protein Classification Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data (100 samples)
  python protein_classification_data_preparation.py --create_sample

  # Generate data with custom number of samples
  python protein_classification_data_preparation.py --create_sample --num_samples 500

  # Process existing data
  python protein_classification_data_preparation.py --input_csv data.csv --output_dir ./processed
""",
    )

    parser.add_argument("--input_csv", type=str, help="Input CSV file with protein variant data")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: $LEARNING_SOURCE_DIR/protein_sequence/data/protein_classification)",
    )

    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset for testing")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )

    args = parser.parse_args()

    try:
        dataset_path = prepare_protein_classification_data(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            create_sample=args.create_sample,
        )

        logger.info("=" * 70)
        logger.info("✅ Data preparation completed successfully")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
