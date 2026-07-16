"""Phase 1-1 (v3): re-run combine_all with the Phase 0-3 dedup patch now
active on the linear stack.

The 07-08 build (`learning_source_20260708_compounds`) inherited a stray
0-3 branch that never landed on the 1-5 stack — so `OrganiX13.parquet`
went out with 310,374 duplicate SMILES (2.33 % of 13,299,737 rows).
The 07-16 rebase (upstream PR #80) brings the dedup into the current
combine_all.py; this script re-runs it against the same raw sources
(unchanged) and writes the deduped parquet to a fresh v3 build tree so
the flawed 07-08 output stays intact for audit.

Layout:
    raw sources : learning_source_20260708_compounds/compounds/data/{zinc20,opv,Fraunhofer-SCAI-llamol}/
    v3 output   : learning_source_20260716_compounds_v3/compounds/organix13/OrganiX13.parquet

Reports pre-dedup vs post-dedup row counts so we can confirm the 310k
duplicates were removed.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase1_1_v3")

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
sys.path.insert(0, str(REPO))

# Raw source data unchanged from the 07-08 build (zinc20 / opv / Fraunhofer-SCAI-llamol).
RAW_ROOT = Path("/lustre/home/matsubara/learning_source_20260708_compounds/compounds")

# New v3 output tree — keeps the flawed 07-08 OrganiX13.parquet on disk for
# audit / regression comparison.
BUILD = Path("/lustre/home/matsubara/learning_source_20260716_compounds_v3")
SAVE_ROOT = BUILD / "compounds" / "organix13"


def main() -> int:
    from molcrawl.data.compounds.dataset.organix13.combine_all import combine_all

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Raw data root (unchanged): %s", RAW_ROOT)
    logger.info("v3 output OrganiX13.parquet dir: %s", SAVE_ROOT)
    logger.info("Sanity: raw sources present:")
    for sub in ("data/zinc20", "data/opv", "data/Fraunhofer-SCAI-llamol"):
        p = RAW_ROOT / sub
        logger.info("  %s exists=%s", p, p.exists())

    logger.info("=== calling combine_all(raw_data_path, save_path) — Phase 0-3 dedup expected ===")
    combine_all(str(RAW_ROOT), str(SAVE_ROOT))

    import pyarrow.parquet as pq
    out = SAVE_ROOT / "OrganiX13.parquet"
    if not out.exists():
        logger.error("expected OrganiX13.parquet at %s but not found", out)
        return 1
    meta = pq.read_metadata(str(out))
    logger.info(
        "[v3 OrganiX13.parquet] rows=%d, cols=%s, size=%.1f MB",
        meta.num_rows, meta.schema.names, out.stat().st_size / 1024 / 1024,
    )

    # Verify dedup post-condition: 0 duplicates in the written parquet.
    tbl = pq.read_table(str(out), columns=["smiles"])
    smi = tbl.column("smiles").to_pylist()
    total, unique = len(smi), len(set(smi))
    dup = total - unique
    logger.info("[v3 dedup check] rows=%d, unique=%d, duplicates=%d (%.4f%%)",
                total, unique, dup, dup * 100.0 / max(total, 1))
    if dup != 0:
        logger.error("DEDUP FAILED — %d duplicates remain, expected 0", dup)
        return 2

    # Compare against the flawed 07-08 baseline for the record.
    old = Path("/lustre/home/matsubara/learning_source_20260708_compounds/compounds/organix13/OrganiX13.parquet")
    if old.exists():
        old_meta = pq.read_metadata(str(old))
        logger.info(
            "[baseline 07-08 non-dedup] rows=%d, delta=%d (removed by dedup)",
            old_meta.num_rows, old_meta.num_rows - total,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
