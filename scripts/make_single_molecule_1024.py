"""Build the (i)/(ii) probe data: one molecule per row, padded to 1024.

Two shapes learn and one does not:

    seq 128, one molecule per row      -> learns (0.68)
    seq 128, packed stream sliced      -> learns (0.67)
    seq 1024, packed stream            -> stuck at 2.43 after 12,000 steps

So the break is tied to the 1024-token block, but two things change together there:
the sequence length itself, and the ~26 unrelated molecules that share the window.
This dataset holds one molecule per row *at* 1024, which keeps the length and drops
the neighbours — if it learns, the window's molecule count is the culprit and
document masking is the fix; if it stalls, the length itself is.

Rows come from the v3 padded build (already one molecule per row at 128) and are
re-padded to 1024. ``attention_mask`` marks the real tokens, so the padding is not
attended to. A subset is enough for a diagnostic and keeps the output ~4 GB.
"""
from __future__ import annotations

import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("single_molecule_1024")

TARGET_LEN = 1024
PAD_ID = 0
CHUNK_ROWS = 50000
OUT_SUFFIX = "_1mol1024"


def write_split(src_path: Path, out_path: Path, split: str, limit: int | None) -> int:
    src = load_from_disk(str(src_path))[split]
    n_src = min(len(src), limit) if limit else len(src)
    out_dir = out_path / split
    out_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = out_dir / "data-00000-of-00001.arrow"

    schema = pa.schema(
        [
            pa.field("input_ids", pa.list_(pa.int32())),
            pa.field("attention_mask", pa.list_(pa.int32())),
        ]
    )
    n_rows = 0
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            for start in range(0, n_src, CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, n_src)
                rows = src[start:stop]
                ids = np.asarray(rows["input_ids"], dtype=np.int32)
                if "attention_mask" in rows:
                    mask = np.asarray(rows["attention_mask"], dtype=np.int32)
                else:
                    mask = (ids != PAD_ID).astype(np.int32)
                assert ids.shape[1] <= TARGET_LEN, f"source rows are longer than {TARGET_LEN}"

                out_ids = np.full((ids.shape[0], TARGET_LEN), PAD_ID, dtype=np.int32)
                out_mask = np.zeros((ids.shape[0], TARGET_LEN), dtype=np.int32)
                out_ids[:, : ids.shape[1]] = ids
                out_mask[:, : mask.shape[1]] = mask

                arrays = []
                for arr in (out_ids, out_mask):
                    values = pa.array(arr.reshape(-1), type=pa.int32())
                    offsets = pa.array(
                        np.arange(0, (arr.shape[0] + 1) * TARGET_LEN, TARGET_LEN, dtype=np.int32)
                    )
                    arrays.append(pa.ListArray.from_arrays(offsets, values))
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                n_rows += ids.shape[0]
                logger.info("  %s: %d/%d rows", split, n_rows, n_src)

    (out_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "citation": "",
                "description": "",
                "features": {
                    name: {"feature": {"dtype": "int32", "_type": "Value"}, "_type": "Sequence"}
                    for name in ("input_ids", "attention_mask")
                },
                "homepage": "",
                "license": "",
                "splits": {
                    split: {
                        "name": split,
                        "num_bytes": int(arrow_path.stat().st_size),
                        "num_examples": int(n_rows),
                        "dataset_name": None,
                    }
                },
            },
            indent=2,
        )
    )
    (out_dir / "state.json").write_text(
        json.dumps(
            {
                "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
                "_fingerprint": f"single_molecule_1024_{split}",
                "_format_columns": None,
                "_format_kwargs": {},
                "_format_type": None,
                "_output_all_columns": False,
                "_split": split,
            },
            indent=2,
        )
    )
    return n_rows


def resolve_paths(src: str | None, out: str | None) -> tuple[Path, Path]:
    if src is None:
        if not os.environ.get("LEARNING_SOURCE_DIR"):
            raise SystemExit(
                "LEARNING_SOURCE_DIR is not set — export it or pass --src explicitly."
            )
        from molcrawl.core.paths import COMPOUNDS_DATASET_DIR_BERT

        src = COMPOUNDS_DATASET_DIR_BERT
    src_path = Path(src)
    out_path = Path(out) if out else src_path.with_name(src_path.name + OUT_SUFFIX)
    return src_path, out_path


def main() -> int:
    parser = ArgumentParser(description="Re-pad one-molecule-per-row data to 1024 tokens")
    parser.add_argument("--src", default=None, help="source dataset (one molecule per row)")
    parser.add_argument("--out", default=None, help=f"output dataset (default: <src>{OUT_SUFFIX})")
    parser.add_argument("--train-rows", type=int, default=1_000_000, help="rows to take from train")
    parser.add_argument("--valid-rows", type=int, default=50_000, help="rows to take from valid")
    args = parser.parse_args()

    src_path, out_path = resolve_paths(args.src, args.out)
    logger.info("source: %s", src_path)
    logger.info("output: %s", out_path)

    counts = {
        "train": write_split(src_path, out_path, "train", args.train_rows),
        "valid": write_split(src_path, out_path, "valid", args.valid_rows),
    }
    (out_path / "dataset_dict.json").write_text(json.dumps({"splits": ["train", "valid"]}, indent=2))

    logger.info("verifying by re-reading ...")
    d = load_from_disk(str(out_path))
    for split, expected in counts.items():
        row = d[split][0]
        ids = np.asarray(row["input_ids"])
        mask = np.asarray(row["attention_mask"])
        assert len(d[split]) == expected, f"{split}: {len(d[split])} != {expected}"
        assert len(ids) == TARGET_LEN, f"{split}: row length {len(ids)}"
        assert int(mask.sum()) == int((ids != PAD_ID).sum()), f"{split}: mask does not match padding"
        logger.info(
            "  %s: %d rows x %d tokens, %d real tokens in row 0 (%.1f%% padding)",
            split, len(d[split]), len(ids), int(mask.sum()),
            100 * (1 - mask.sum() / TARGET_LEN),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
