#!/usr/bin/env python3
"""
Script to perform random sampling from ClinVar data

Usage example:
1. Random extraction from existing CSV files:
   python extract_random_clinvar_samples.py --input_csv clinvar_sequences.csv --output_csv random_2000.csv --num_samples 2000

2. Sequence generation by random sampling directly from the dataset:
   python extract_random_clinvar_samples.py --ref_fasta GCA_000001405.28_GRCh38.p13_genomic.fna --output_csv random_2000.csv --num_samples 2000 --flank 64
"""

import argparse
import gzip
import os
import re
import shutil

import numpy as np
import pandas as pd
from datasets import load_dataset
from pyfaidx import Fasta


def classify_clinical_significance(clin_sig):
    """Classify Clinical Significance as negative/positive"""
    if pd.isna(clin_sig) or clin_sig is None:
        return None

    clin_sig_str = str(clin_sig).lower()

    # Negative (benign) pattern
    benign_patterns = [
        "benign",
        "likely_benign",
        "likely benign",
        "protective",
        "not provided",  # not provided is treated as negative as neutral
    ]

    # Positive (pathogenic) pattern
    pathogenic_patterns = [
        "pathogenic",
        "likely_pathogenic",
        "likely pathogenic",
        "pathogenic/likely_pathogenic",
        "drug response",
        "association",
        "risk factor",
        "affects",
    ]

    for pattern in pathogenic_patterns:
        if pattern in clin_sig_str:
            return "pathogenic"

    for pattern in benign_patterns:
        if pattern in clin_sig_str:
            return "benign"

    # Others (uncertain significance, etc.) are treated as negative as neutral.
    return "benign"


def balanced_sampling(df, num_samples, clin_sig_col="ClinicalSignificance", seed=42):
    """Balanced sampling with half negative and half positive"""
    np.random.seed(seed)

    # Classify clinical significance
    df = df.copy()
    df["classification"] = df[clin_sig_col].apply(classify_clinical_significance)

    # Check the classification results
    print("Original classification distribution:")
    class_counts = df["classification"].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # Separate negative and positive data
    benign_df = df[df["classification"] == "benign"].copy()
    pathogenic_df = df[df["classification"] == "pathogenic"].copy()

    print(f"\nAvailable samples: Benign={len(benign_df)}, Pathogenic={len(pathogenic_df)}")

    # Sample half from each class
    samples_per_class = num_samples // 2

    # Check number of available samples
    if len(benign_df) < samples_per_class:
        print(f"Warning: Not enough benign samples ({len(benign_df)} < {samples_per_class})")
        samples_per_class = min(len(benign_df), len(pathogenic_df))
        print(f"Adjusting to {samples_per_class} samples per class")

    if len(pathogenic_df) < samples_per_class:
        print(f"Warning: Not enough pathogenic samples ({len(pathogenic_df)} < {samples_per_class})")
        samples_per_class = min(len(benign_df), len(pathogenic_df))
        print(f"Adjusting to {samples_per_class} samples per class")

    # Execute sampling
    sampled_benign = benign_df.sample(n=samples_per_class, random_state=seed)
    sampled_pathogenic = pathogenic_df.sample(n=samples_per_class, random_state=seed + 1)

    # combine and shuffle
    balanced_df = pd.concat([sampled_benign, sampled_pathogenic], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=seed + 2).reset_index(drop=True)

    print("\nBalanced sampling result:")
    final_counts = balanced_df["classification"].value_counts()
    for cls, count in final_counts.items():
        print(f"  {cls}: {count}")

    return balanced_df


def build_chrom_mapping(ref_genome):
    """Build chromosome mapping"""
    headers = [ref_genome[seq].long_name for seq in ref_genome.keys()]
    mapping = {}
    for h in headers:
        m = re.search(r"^(CM\d+\.\d+).*chromosome (\w+)", h)
        if m:
            seq_id = m.group(1)
            chrom = m.group(2)
            if chrom.lower().startswith("mito"):
                chrom = "MT"
            mapping[chrom] = seq_id
    return mapping


def get_sequences(ref_genome, mapping, chrom, pos, ref, alt, flank=64):
    """Get reference and variant sequences"""
    seq_id = mapping[str(chrom)]
    start = pos - flank
    end = pos + flank
    ref_seq = ref_genome[seq_id][start - 1 : end].seq.upper()

    center_base = ref_seq[flank]
    if center_base != ref.upper():
        print(f"Warning: reference mismatch at {chrom}:{pos}, expected {ref}, got {center_base}")

    seq_list = list(ref_seq)
    seq_list[flank] = alt.upper()
    var_seq = "".join(seq_list)

    return ref_seq, var_seq


def extract_from_csv(csv_path, num_samples, output_path, seed=42):
    """Extract a balanced sample from the CSV file (half negative and half positive)"""
    print(f"Reading CSV file: {csv_path}")

    # Read CSV file (auto detect delimiter)
    if csv_path.endswith(".gz"):
        # Try tab delimiter first
        try:
            df = pd.read_csv(csv_path, compression="gzip", sep="\t", low_memory=False)
        except (ValueError, pd.errors.ParserError):
            df = pd.read_csv(csv_path, compression="gzip", sep=",", low_memory=False)
    else:
        # Try tab delimiter first
        try:
            df = pd.read_csv(csv_path, sep="\t", low_memory=False)
            # If there is only one column, try comma separation
            if len(df.columns) == 1:
                df = pd.read_csv(csv_path, sep=",", low_memory=False)
        except (ValueError, pd.errors.ParserError):
            df = pd.read_csv(csv_path, sep=",", low_memory=False)

    print(f"Total records: {len(df)}")

    # Search the Clinical Significance column
    clin_sig_col = None
    for col in [
        "ClinicalSignificance",
        "clinical_significance",
        "clin_sig",
        "significance",
    ]:
        if col in df.columns:
            clin_sig_col = col
            break

    if clin_sig_col:
        print(f"Original Clinical Significance distribution ({clin_sig_col}):")
        clin_sig_counts = df[clin_sig_col].value_counts()
        for sig, count in clin_sig_counts.head(10).items():
            print(f"  {sig}: {count}")

    # Execute balanced sampling
    if num_samples > len(df):
        print(f"Requested samples ({num_samples}) > available records ({len(df)})")
        sampled_df = df.copy()
    else:
        if clin_sig_col:
            sampled_df = balanced_sampling(df, num_samples, clin_sig_col, seed=seed)
        else:
            print("Warning: No Clinical Significance column found, using random sampling")
            np.random.seed(seed)
            sampled_df = df.sample(n=num_samples, random_state=seed)

    print(f"Sampled {len(sampled_df)} records")

    # Output (The original file is comma-separated, so output in comma-separated format)
    sampled_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return sampled_df


def extract_from_dataset(ref_fasta, output_csv, num_samples, flank=64, seed=42):
    """Sequence generation by balanced sampling directly from the dataset (half negative, half positive)"""
    print("Loading ClinVar dataset from HuggingFace...")
    dataset = load_dataset("gonzalobenegas/clinvar")
    df = dataset["test"].to_pandas()

    print(f"Total dataset size: {len(df)}")

    if len(df) < num_samples:
        print(f"Warning: Dataset has only {len(df)} samples, less than requested {num_samples}")
        num_samples = len(df)

    # Execute balanced sampling
    if "clin_sig" in df.columns:
        sampled_df = balanced_sampling(df, num_samples, "clin_sig", seed=seed)
    else:
        print("Warning: No clin_sig column found, using random sampling")
        np.random.seed(seed)
        sampled_df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)

    print(f"Sampled {len(sampled_df)} variants")

    # Check the distribution of clinical significance
    if "clin_sig" in sampled_df.columns:
        print("\nClinical Significance distribution in sample:")
        clin_sig_counts = sampled_df["clin_sig"].value_counts()
        for sig, count in clin_sig_counts.items():
            print(f"  {sig}: {count}")

    # Automatic file format support
    ref_fasta_path = ref_fasta
    if ref_fasta_path.endswith(".gz"):
        uncompressed_path = ref_fasta_path.replace(".gz", "")
        if os.path.exists(uncompressed_path):
            print(f"Using uncompressed FASTA file: {uncompressed_path}")
            ref_fasta_path = uncompressed_path
        else:
            print(f"Uncompressing {ref_fasta_path} for compatibility...")
            with gzip.open(ref_fasta_path, "rb") as f_in:
                with open(uncompressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            ref_fasta_path = uncompressed_path
            print(f"Uncompressed to: {ref_fasta_path}")

    # Load reference genome
    print("Loading reference genome...")
    ref_genome = Fasta(ref_fasta_path)
    mapping = build_chrom_mapping(ref_genome)

    # create array
    print("Generating sequences...")
    records = []
    for i, row in sampled_df.iterrows():
        try:
            ref_seq, var_seq = get_sequences(
                ref_genome,
                mapping,
                row["chrom"],
                row["pos"],
                row["ref"],
                row["alt"],
                flank=flank,
            )

            record = {
                "chrom": row["chrom"],
                "pos": row["pos"],
                "ref": row["ref"],
                "alt": row["alt"],
                "reference_sequence": ref_seq,
                "variant_sequence": var_seq,
            }

            # Add ClinicalSignificance
            if "clin_sig" in row:
                record["ClinicalSignificance"] = row["clin_sig"]
            else:
                record["ClinicalSignificance"] = None

            records.append(record)

            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(sampled_df)} variants")

        except Exception as e:
            print(f"Error at {row['chrom']}:{row['pos']} - {e}")
            continue

    # Save to CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)

    print(f"Saved {len(df_out)} variants with sequences to {output_csv}")

    # final statistics
    if "ClinicalSignificance" in df_out.columns:
        print("\nFinal Clinical Significance distribution:")
        final_counts = df_out["ClinicalSignificance"].value_counts()
        for sig, count in final_counts.items():
            print(f"  {sig}: {count}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Extract random samples from ClinVar data")
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Input CSV file (for extraction from existing file)",
    )
    parser.add_argument(
        "--ref_fasta",
        type=str,
        help="Reference FASTA file (for extraction from dataset)",
    )
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to extract (default: 2000)",
    )
    parser.add_argument(
        "--flank",
        type=int,
        default=64,
        help="Flank size for sequence extraction (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Check input mode
    if args.input_csv and args.ref_fasta:
        print("Error: Specify either --input_csv OR --ref_fasta, not both")
        return
    elif args.input_csv:
        print("Mode: Extracting from existing CSV file")
        extract_from_csv(args.input_csv, args.num_samples, args.output_csv, args.seed)
    elif args.ref_fasta:
        print("Mode: Extracting from dataset with sequence generation")
        extract_from_dataset(args.ref_fasta, args.output_csv, args.num_samples, args.flank, args.seed)
    else:
        print("Error: Specify either --input_csv or --ref_fasta")
        parser.print_help()


if __name__ == "__main__":
    main()
