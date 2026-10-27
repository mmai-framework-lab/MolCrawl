"""Measure the protein_sequence (UniRef50) corpus provenance, stage by stage.

protein-uniref-order-2026-10-27: confirm what release was used and how the
38,794,121 figure is produced, by counting the *untouched* download and every
processing stage from files only (no GPU, no training). Emits a one-row-per-stage
TSV (directive section 7) and a JSON record.

Stages measured (each: entrance count -> exit count, dropped):
  1. download        uniref50.fasta  -- FASTA entries (lines starting with '>')
  2. fasta_to_raw    raw_files/*.raw -- one sequence per line
  3. tokenize        parquet_files/*.parquet -- one row per sequence
  4. pack+split      training_ready_hf_dataset/{train,valid,test} -- packed windows

Also: md5 of uniref50.fasta.gz (to answer what the recorded md5 is a hash of),
residue totals and mean residue length computed directly from the source FASTA,
and the count every stage is cross-checked against.

Run (on a compute node):
  python scripts/audit_protein_corpus_provenance.py \
      --data-dir <.../protein_sequence> --out <out_dir>
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path


def count_fasta(fasta: Path):
    """Entries (header lines) and residue chars, streamed. No full load."""
    entries = 0
    residues = 0
    with open(fasta, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                entries += 1
            else:
                residues += len(line.strip())
    return entries, residues


def count_raw(raw_dir: Path):
    """raw_files/*.raw: one sequence per line. Lines and residue chars."""
    files = sorted(raw_dir.glob("*.raw"))
    lines = 0
    residues = 0
    unreadable = []
    for p in files:
        try:
            with open(p, "r") as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        lines += 1
                        residues += len(s)
        except OSError as e:
            unreadable.append((p.name, str(e)))
    return len(files), lines, residues, unreadable


def count_parquet(parquet_dir: Path):
    """Rows across parquet_files/*.parquet via footers only (pyarrow)."""
    import pyarrow.parquet as pq

    files = sorted(parquet_dir.glob("*.parquet"))
    rows = 0
    unreadable = []
    for p in files:
        try:
            rows += pq.ParquetFile(p).metadata.num_rows
        except Exception as e:  # noqa: BLE001 - record and continue
            unreadable.append((p.name, str(e)))
    return len(files), rows, unreadable


def count_splits(ds_dir: Path):
    """Packed-window rows per split, counted from the arrow shards directly."""
    import pyarrow.ipc as ipc

    out = {}
    for split in ("train", "valid", "test"):
        sdir = ds_dir / split
        if not sdir.is_dir():
            continue
        n = 0
        for af in sorted(sdir.glob("*.arrow")):
            with ipc.open_stream(open(af, "rb")) as reader:
                for batch in reader:
                    n += batch.num_rows
        out[split] = n
    return out


def md5_of(path: Path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="protein_sequence tree")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--special-per-seq", type=int, default=2, help="<cls>+<eos>")
    args = ap.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fasta = data / "UniRef50" / "uniref50.fasta"
    gz = data / "UniRef50" / "uniref50.fasta.gz"
    raw_dir = data / "raw_files"
    parquet_dir = data / "parquet_files"
    ds_dir = data / "training_ready_hf_dataset"
    tc_pkl = data / "token_counts.pkl"

    rec = {"data_dir": str(data)}

    # ---- stage 1: untouched download (FASTA) ----
    rec["fasta_path"] = str(fasta)
    rec["fasta_bytes"] = fasta.stat().st_size if fasta.exists() else None
    if fasta.exists():
        ent, res = count_fasta(fasta)
        rec["fasta_entries"] = ent
        rec["fasta_residues"] = res
    else:
        rec["fasta_entries"] = rec["fasta_residues"] = None

    # ---- md5 of the .gz (what the recorded md5 is a hash of) ----
    rec["gz_path"] = str(gz)
    rec["gz_bytes"] = gz.stat().st_size if gz.exists() else None
    rec["gz_md5"] = md5_of(gz) if gz.exists() else None

    # ---- stage 2: fasta_to_raw ----
    nraw, rawlines, rawres, rawbad = count_raw(raw_dir)
    rec["raw_files"] = nraw
    rec["raw_sequences"] = rawlines
    rec["raw_residues"] = rawres
    rec["raw_unreadable"] = rawbad

    # ---- stage 3: tokenize (parquet) ----
    npq, pqrows, pqbad = count_parquet(parquet_dir)
    rec["parquet_files"] = npq
    rec["parquet_rows"] = pqrows
    rec["parquet_unreadable"] = pqbad

    # token_counts.pkl cross-check (len + sum)
    if tc_pkl.exists():
        import pickle

        tc = pickle.load(open(tc_pkl, "rb"))
        rec["token_counts_len"] = len(tc)
        rec["token_counts_sum"] = int(sum(tc))
        rec["residues_from_token_counts"] = int(sum(tc)) - args.special_per_seq * len(tc)

    # ---- stage 4: pack + split ----
    splits = count_splits(ds_dir)
    rec["splits_windows"] = splits
    rec["windows_total"] = sum(splits.values()) if splits else None

    # ---- mean residue length (directive section 6) ----
    n = rec.get("fasta_entries") or rec.get("raw_sequences")
    if n and rec.get("fasta_residues"):
        rec["mean_residue_len_from_fasta"] = rec["fasta_residues"] / n
    if n and rec.get("raw_residues"):
        rec["mean_residue_len_from_raw"] = rec["raw_residues"] / n

    with open(out / "protein-corpus-provenance.json", "w") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)

    # ---- stage TSV (directive section 7): one row per stage ----
    ent = rec.get("fasta_entries")
    rows = [
        ("1_download_fasta_entries", "", ent, ""),
        ("2_fasta_to_raw_lines", ent, rec.get("raw_sequences"), _diff(ent, rec.get("raw_sequences"))),
        ("3_tokenize_parquet_rows", rec.get("raw_sequences"), rec.get("parquet_rows"),
         _diff(rec.get("raw_sequences"), rec.get("parquet_rows"))),
        ("4_pack_split_windows_total", rec.get("parquet_rows"), rec.get("windows_total"), "packing (not a drop)"),
    ]
    for sp in ("train", "valid", "test"):
        if sp in splits:
            rows.append((f"4_{sp}_windows", "", splits[sp], ""))
    with open(out / "protein-corpus-stages.tsv", "w") as fh:
        fh.write("stage\tin_count\tout_count\tdropped\n")
        for name, a, b, d in rows:
            fh.write(f"{name}\t{a}\t{b}\t{d}\n")

    print(json.dumps(rec, indent=2, ensure_ascii=False))
    print(f"\nwrote {out/'protein-corpus-provenance.json'}")
    print(f"wrote {out/'protein-corpus-stages.tsv'}")


def _diff(a, b):
    if a is None or b is None:
        return ""
    return a - b


if __name__ == "__main__":
    sys.exit(main())
