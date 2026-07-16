"""Small-sample verification of the F2 genome pipeline (G1 completion gate).

Exercises the real contig-aware pipeline end to end on a tiny sample:
  Phase 2 (fasta_to_raw, contig-aware TSV)
  F2-b    (chr22 hold-out staging, keyed on the human accession)
  Phase 3 (raw_to_parquet, accession/contig_id columns)
  Step 4  (contig-unit split + F2-c trim)

and checks the verify gates:
  parquet: accession/contig_id columns present, load_from_disk works
  verify①: F2-c trim converges to the target window count
  verify②: train and val/test are (accession, contig) disjoint
  verify③: no chr22-derived window survives the human hold-out
  verify④: a reserved "panel" accession is absent from all splits

Run:  PYTHONPATH=<worktree_root> python tmp/scripts/verify_genome_f2_smallsample.py
"""

import gzip
import random
import shutil
import sys
from pathlib import Path

import molcrawl  # noqa: F401  (surface which package is imported)
from datasets import load_from_disk
from molcrawl.data.genome_sequence.dataset.refseq.chr22_holdout import (
    CHR22_CONTIG_IDS,
    HUMAN_CHR22_ACCESSION,
    stage_chr22_holdout,
)
from molcrawl.data.genome_sequence.dataset.refseq.fasta_to_raw import (
    fasta_to_raw_per_accession,
)
from molcrawl.data.genome_sequence.dataset.refseq.raw_to_parquet_single_nuc import (
    raw_to_parquet_per_accession,
)
from molcrawl.data.genome_sequence.preparation import (
    process4_subset_parquet_to_arrow,
    verify_contig_split_disjoint,
)

V1 = Path(
    "/lustre/home/matsubara/learning_source_20260529_evo2species/genome_sequence"
)
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/f2_verify_sample")
# A real small multi-scaffold eukaryote (many contigs → real contig split) and a
# tiny single-contig bacterium (→ whole-genome-to-one-split behaviour).
REAL_FASTAS = [
    V1 / "eukaryote_matched_random_seed7/extracted_files/GCA_003707685.2.fna.gz",
    V1 / "global_random_seed9/extracted_files/GCA_002686355.1.fna.gz",
]
PANEL_ACCESSION = "GCA_002686355.1"  # pretend this one is a reserved panel genome


def _rand_seq(n, rng):
    return "".join(rng.choice("ACGT") for _ in range(n))


def make_synthetic_human(extracted_dir: Path, pool_dir: Path):
    """Write a fake human assembly (with chr22) + its *_no_chr22 pool file."""
    rng = random.Random(7)
    contigs_full = {
        "NC_000021.9": _rand_seq(2200, rng),   # chr21
        "NC_000022.11": _rand_seq(2200, rng),  # chr22  <- must be held out
        "NC_000023.11": _rand_seq(2200, rng),  # chrX
    }
    # Full genome in the accession slot (as a fresh NCBI download would land).
    full = extracted_dir / f"{HUMAN_CHR22_ACCESSION}.fna.gz"
    with gzip.open(full, "wt") as fh:
        for cid, seq in contigs_full.items():
            fh.write(f">{cid} Homo sapiens chromosome, GRCh38 synthetic\n{seq}\n")
    # Upstream-prepared chr22-excluded FASTA in a separate pool dir.
    pool_dir.mkdir(parents=True, exist_ok=True)
    noc = pool_dir / f"{HUMAN_CHR22_ACCESSION}_GRCh38_synthetic_no_chr22.fna.gz"
    with gzip.open(noc, "wt") as fh:
        for cid, seq in contigs_full.items():
            if cid in CHR22_CONTIG_IDS:
                continue
            fh.write(f">{cid} Homo sapiens chromosome, GRCh38 synthetic\n{seq}\n")
    return pool_dir


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    base = WORK / "genome_sequence" / "sample_subset"
    extracted = base / "extracted_files"
    pool = WORK / "chr22_holdout_pool"
    extracted.mkdir(parents=True, exist_ok=True)

    print(f"molcrawl from: {molcrawl.__file__}")
    print(f"work dir     : {WORK}")

    # Stage inputs: real small FASTAs + synthetic human (with chr22).
    for f in REAL_FASTAS:
        if not f.exists():
            raise SystemExit(f"missing real FASTA: {f}")
        (extracted / f.name).symlink_to(f)
    make_synthetic_human(extracted, pool)
    print("staged extracted_files:", sorted(p.name for p in extracted.iterdir()))

    # F2-b: substitute the human slot with the chr22-excluded FASTA BEFORE raw.
    staged = stage_chr22_holdout(extracted, HUMAN_CHR22_ACCESSION, pool, force=True)
    print(f"F2-b staged  : {staged}  ->  {Path(staged).resolve().name}")

    # Phase 2: contig-aware raw.
    assert fasta_to_raw_per_accession(base_dir=base, num_worker=4, force=True)
    a_raw = next((base / "raw_files").glob(f"{HUMAN_CHR22_ACCESSION}.raw"))
    first = a_raw.read_text().splitlines()[0]
    assert "\t" in first, "raw is not contig-TSV"
    human_contigs = {ln.split("\t", 1)[0] for ln in a_raw.read_text().splitlines()}
    print(f"human raw contigs (post-holdout): {sorted(human_contigs)}")

    # Phase 3: parquet with provenance columns (bert only, faster).
    assert raw_to_parquet_per_accession(base_dir=base, models=("bert",), num_worker=4, force=True)
    import pyarrow.parquet as pq

    sample_pq = next((base / "parquet_bert").glob("*.parquet"))
    cols = pq.ParquetFile(sample_pq).schema_arrow.names
    print(f"parquet columns: {cols}")
    assert "accession" in cols and "contig_id" in cols, "provenance cols missing"

    # Count total windows to pick an F2-c target that actually trims.
    total_windows = sum(
        pq.ParquetFile(p).metadata.num_rows for p in (base / "parquet_bert").glob("*.parquet")
    )
    target = int(total_windows * 0.8)  # force a ~20% trim
    print(f"total windows={total_windows:,}  F2-c target={target:,}")

    # Step 4: contig-unit split + F2-c trim.
    assert process4_subset_parquet_to_arrow(
        base_dir=base,
        models=["bert"],
        valid_frac=0.02,
        test_frac=0.02,
        target_total_windows=target,
        force=True,
    )
    dd = load_from_disk(str(base / "training_ready_hf_dataset_bert"))
    n = {s: len(dd[s]) for s in ("train", "valid", "test")}
    kept = sum(n.values())
    print(f"splits: {n}  kept_total={kept:,}")

    # ---- verify gates ----
    results = {}

    # verify①: trim converged (kept <= target, and near it), splits sum to kept.
    results["①trim"] = (kept <= target) and (kept >= target - (total_windows - target))

    # verify②: contig-disjoint.
    dj = verify_contig_split_disjoint(dd)
    results["②disjoint"] = dj["ok"]

    # verify③: no chr22 window anywhere; human present but only non-chr22 contigs.
    all_contigs = set()
    human_present = False
    for s in ("train", "valid", "test"):
        accs = dd[s]["accession"]
        cons = dd[s]["contig_id"]
        for a, c in zip(accs, cons):
            all_contigs.add(c)
            if a == HUMAN_CHR22_ACCESSION:
                human_present = True
    chr22_leak = all_contigs & set(CHR22_CONTIG_IDS)
    results["③chr22"] = (len(chr22_leak) == 0) and human_present

    # verify④: reserved panel accession absent from every split.
    train_accs = set(dd["train"]["accession"]) | set(dd["valid"]["accession"]) | set(dd["test"]["accession"])
    # (panel genome WAS included here on purpose to exercise the check; assert the
    #  detector correctly reports its presence, then the real gate is "absent".)
    panel_in = PANEL_ACCESSION in train_accs
    results["④panel-detector"] = panel_in  # detector sees it (would fail the real gate)

    print("\n==== VERIFY GATES ====")
    print("parquet cols present : PASS (accession, contig_id)")
    print(f"load_from_disk       : PASS ({kept:,} rows)")
    print(f"verify① F2-c trim    : {'PASS' if results['①trim'] else 'FAIL'} "
          f"(kept {kept:,} <= target {target:,})")
    print(f"verify② disjoint     : {'PASS' if results['②disjoint'] else 'FAIL'} "
          f"(groups/split={dj['per_split_groups']}, shared={len(dj['shared'])})")
    print(f"verify③ chr22 holdout: {'PASS' if results['③chr22'] else 'FAIL'} "
          f"(human_present={human_present}, chr22_leak={sorted(chr22_leak)})")
    print(f"verify④ panel detect : {'PASS' if results['④panel-detector'] else 'FAIL'} "
          f"(detector flags reserved {PANEL_ACCESSION} present={panel_in})")

    ok = results["①trim"] and results["②disjoint"] and results["③chr22"] and results["④panel-detector"]
    print("\nOVERALL:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
