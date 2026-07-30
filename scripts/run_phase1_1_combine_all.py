"""Phase 1-1: run combine_all.py against the new 20260708 build area.

Reads 8 raw source parquets from `learning_source_20260708_compounds/compounds/data/`
and writes a single OrganiX13.parquet to the same tree's organix13/ subdir. Uses
the fixed combine_all.py (Phase 0-3 dedup) — no silent-skip risk since Phase 0-2
verified the current code already raises FileNotFoundError.

Reports:
- per-source row count as logged
- final combined row count (pre-dedup vs post-dedup)
- ZINC20 inclusion confirmation
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase1_1")

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
sys.path.insert(0, str(REPO))

BUILD = Path("/lustre/home/matsubara/learning_source_20260708_compounds")
RAW_ROOT = BUILD / "compounds"        # combine_all reads $RAW_ROOT/data/...
SAVE_ROOT = BUILD / "compounds" / "organix13"


def main() -> int:
    from molcrawl.data.compounds.dataset.organix13.combine_all import combine_all

    logger.info("Build area: %s", BUILD)
    logger.info("Raw data root: %s", RAW_ROOT)
    logger.info("Output OrganiX13.parquet dir: %s", SAVE_ROOT)

    logger.info("=== calling combine_all(raw_data_path, save_path) ===")
    combine_all(str(RAW_ROOT), str(SAVE_ROOT))

    # Post-check: report OrganiX13.parquet contents
    import pyarrow.parquet as pq
    out = SAVE_ROOT / "OrganiX13.parquet"
    if not out.exists():
        logger.error("expected OrganiX13.parquet at %s but not found", out)
        return 1
    meta = pq.read_metadata(str(out))
    logger.info("[OrganiX13.parquet] rows=%d, cols=%s, size=%.1f MB",
                meta.num_rows, meta.schema.names, out.stat().st_size / 1024 / 1024)
    # ZINC20 signature: sample a few SMILES and check for typical ZINC-like structures.
    tbl = pq.read_table(str(out), columns=["smiles"])
    smi_head = tbl.column("smiles").to_pylist()[:5]
    logger.info("[first 5 SMILES]:")
    for s in smi_head:
        logger.info("    %s", s[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
