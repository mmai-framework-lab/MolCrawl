#!/usr/bin/env python3
"""
Prepare ClinVar benchmark sequences from HuggingFace dataset and GRCh38 reference.

Usage:
  python prepare_clinvar_sequences.py \
      --ref_fasta GCA_000001405.28_GRCh38.p13_genomic.fna.gz \
      --output_file clinvar_sequences.csv \
      --flank 64 \
      --max_samples 100
"""

import argparse
import gzip
import os
import re
import shutil

import pandas as pd
from datasets import load_dataset
from pyfaidx import Fasta


def build_chrom_mapping(ref_genome):
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


def main():
    parser = argparse.ArgumentParser(description="Prepare ClinVar benchmark sequences")
    parser.add_argument(
        "--ref_fasta",
        type=str,
        required=True,
        help="Path to GRCh38 genomic FASTA (e.g. GCA_000001405.28_GRCh38.p13_genomic.fna or .fna.gz)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="clinvar_sequences.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--flank",
        type=int,
        default=64,
        help="Number of bp to take on each side (default=64 → 128bp window)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples to process (default: all)",
    )

    args = parser.parse_args()

    # Automatic file format support
    ref_fasta_path = args.ref_fasta
    if ref_fasta_path.endswith(".gz"):
        # For .gz files, check if there is an unzipped version
        uncompressed_path = ref_fasta_path.replace(".gz", "")
        if os.path.exists(uncompressed_path):
            print(f"Using uncompressed FASTA file: {uncompressed_path}")
            ref_fasta_path = uncompressed_path
        else:
            # Temporarily extract if there is no expanded version
            print(f"Uncompressing {ref_fasta_path} for compatibility...")
            with gzip.open(ref_fasta_path, "rb") as f_in:
                with open(uncompressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            ref_fasta_path = uncompressed_path
            print(f"Uncompressed to: {ref_fasta_path}")

    dataset = load_dataset("gonzalobenegas/clinvar")
    df = dataset["test"].to_pandas()

    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Processing only first {args.max_samples} variants for testing")

    # debug: check available columns
    print(f"Available columns in dataset: {df.columns.tolist()}")

    # Check the existence of ClinicalSignificance column
    clinical_significance_col = None
    possible_names = [
        "ClinicalSignificance",
        "clinical_significance",
        "clin_sig",
        "clnsig",
        "significance",
    ]
    for col_name in possible_names:
        if col_name in df.columns:
            clinical_significance_col = col_name
            print(f"Found clinical significance column: {col_name}")
            break

    if clinical_significance_col is None:
        print("Warning: Clinical significance column not found. Available columns:")
        for col in df.columns:
            print(f"  - {col}")
    else:
        # Display distribution of clinical significance
        print("\nClinical significance distribution in source data:")
        clin_sig_counts = df[clinical_significance_col].value_counts()
        for sig, count in clin_sig_counts.head(10).items():  # Display top 10
            print(f"  {sig}: {count}")

    ref_genome = Fasta(ref_fasta_path)
    mapping = build_chrom_mapping(ref_genome)

    records = []
    for _i, row in df.iterrows():
        try:
            ref_seq, var_seq = get_sequences(
                ref_genome,
                mapping,
                row["chrom"],
                row["pos"],
                row["ref"],
                row["alt"],
                flank=args.flank,
            )

            # Basic mutation information
            record = {
                "chrom": row["chrom"],
                "pos": row["pos"],
                "ref": row["ref"],
                "alt": row["alt"],
                "reference_sequence": ref_seq,
                "variant_sequence": var_seq,
            }

            # Add label column if there is one (for backward compatibility)
            if "label" in row:
                record["label"] = row["label"]

            # Add ClinicalSignificance(if any)
            if clinical_significance_col:
                record["ClinicalSignificance"] = row[clinical_significance_col]
            else:
                record["ClinicalSignificance"] = None

            records.append(record)

        except Exception as e:
            print(f"Error at {row['chrom']}:{row['pos']} - {e}")
            continue

    df_out = pd.DataFrame(records)
    df_out.to_csv(args.output_file, index=False)

    # Output of statistics information
    print(f"Saved {len(df_out)} variants with sequences → {args.output_file}")

    if clinical_significance_col:
        print("\nClinical Significance distribution:")
        significance_counts = df_out["ClinicalSignificance"].value_counts()
        for sig, count in significance_counts.items():
            print(f"  {sig}: {count}")

        # Also shows pathogenic/benign distribution
        pathogenic_count = df_out["ClinicalSignificance"].str.contains("pathogenic", case=False, na=False).sum()
        benign_count = df_out["ClinicalSignificance"].str.contains("benign", case=False, na=False).sum()
        uncertain_count = df_out["ClinicalSignificance"].str.contains("uncertain", case=False, na=False).sum()

        print("\nSummary:")
        print(f"  Pathogenic variants: {pathogenic_count}")
        print(f"  Benign variants: {benign_count}")
        print(f"  Uncertain significance: {uncertain_count}")
        print(f"  Other/Missing: {len(df_out) - pathogenic_count - benign_count - uncertain_count}")
    else:
        print("\nNo ClinicalSignificance data available in the dataset.")


if __name__ == "__main__":
    main()
