"""Numeric-equivalence check: streaming contig-unit split == in-memory oracle.

Builds synthetic per-accession parquet files (schema-faithful: accession +
contig_id, mirroring Phase 3 output — one accession per file, several contigs,
uneven window counts), then asserts that _contig_unit_split_streaming returns
byte-identical train/valid/test row indices and stats to the in-memory
_contig_unit_split, across several seeds and with / without an F2-c trim target.

The in-memory oracle is fed the SAME concatenation order load_dataset would use
(sorted path list, rows in file order) so the two index spaces coincide.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from molcrawl.data.genome_sequence.preparation import (
    _contig_unit_split,
    _contig_unit_split_streaming,
)

WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/eq_split_sample")


def build_parquets(work: Path, rng: np.random.RandomState):
    """Write per-accession parquet files; return (sorted_paths, acc_col, con_col).

    acc_col / con_col are the accession / contig_id sequences in the exact order
    load_dataset(sorted paths) would concatenate them (for the oracle input).
    """
    pdir = work / "parquet_bert"
    if pdir.exists():
        import shutil
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)

    # A spread of assemblies: some multi-contig eukaryotes, some single-contig.
    specs = []
    for i in range(12):
        acc = f"GCA_{1000000 + i:07d}.{1 + i % 3}"
        n_contigs = int(rng.randint(1, 6))
        contigs = []
        for c in range(n_contigs):
            cid = f"NC_{100000 + i * 10 + c:06d}.{1 + c}"
            nwin = int(rng.randint(1, 400))  # uneven window counts per contig
            contigs.append((cid, nwin))
        specs.append((acc, contigs))

    # Include a deliberate single-window contig and a large one to stress packing.
    specs.append(("GCA_9999999.1", [("NC_777777.1", 1)]))
    specs.append(("GCA_8888888.1", [("NC_666666.1", 5000)]))

    for acc, contigs in specs:
        rows_acc, rows_con = [], []
        for cid, nwin in contigs:
            rows_acc.extend([acc] * nwin)
            rows_con.extend([cid] * nwin)
        tbl = pa.table(
            {
                "input_ids": pa.array([[0]] * len(rows_acc), type=pa.list_(pa.int32())),
                "accession": pa.array(rows_acc, type=pa.string()),
                "contig_id": pa.array(rows_con, type=pa.string()),
            }
        )
        pq.write_table(tbl, pdir / f"{acc}.parquet", compression="snappy")

    sorted_paths = sorted(str(p) for p in pdir.glob("*.parquet"))
    acc_col, con_col = [], []
    for p in sorted_paths:
        t = pq.read_table(p, columns=["accession", "contig_id"])
        acc_col.extend(t.column("accession").to_pylist())
        con_col.extend(t.column("contig_id").to_pylist())
    return sorted_paths, acc_col, con_col


def main():
    ncase = 0
    for seed in (42, 7, 123):
        for frac in (0.005, 0.02, 0.05):
            for trim in (None, "trim"):
                rng = np.random.RandomState(seed * 31 + int(frac * 1000))
                sorted_paths, acc_col, con_col = build_parquets(WORK, rng)
                n_total = len(acc_col)
                tgt = int(n_total * 0.8) if trim else None

                a_tr, a_va, a_te, a_st = _contig_unit_split(
                    accessions=acc_col, contigs=con_col,
                    valid_frac=frac, test_frac=frac,
                    target_total_windows=tgt, seed=seed,
                )
                b_tr, b_va, b_te, b_st = _contig_unit_split_streaming(
                    parquet_files=sorted_paths,
                    valid_frac=frac, test_frac=frac,
                    target_total_windows=tgt, seed=seed,
                )
                assert np.array_equal(a_tr, b_tr), (seed, frac, trim, "train")
                assert np.array_equal(a_va, b_va), (seed, frac, trim, "valid")
                assert np.array_equal(a_te, b_te), (seed, frac, trim, "test")
                assert a_st == b_st, (seed, frac, trim, a_st, b_st)
                # sanity: partition covers exactly the kept windows, disjoint.
                allk = np.concatenate([b_tr, b_va, b_te])
                assert len(np.unique(allk)) == len(allk)
                assert len(allk) == a_st["n_kept_windows"]
                ncase += 1
                print(
                    f"seed={seed} frac={frac} trim={bool(trim)}: PASS  "
                    f"train={len(b_tr)} valid={len(b_va)} test={len(b_te)} "
                    f"kept={a_st['n_kept_windows']}/{a_st['n_total_windows']} "
                    f"groups={a_st['n_groups']}"
                )
    print(f"\nALL {ncase} CASES: streaming == in-memory oracle (exact index + stats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
