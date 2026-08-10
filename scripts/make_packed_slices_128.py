"""Build the V2 arm's data: the packed 1024 blocks cut into 128-token rows.

The cross-document probe compares three shapes:

    V1  v3 padded      seq 128, one molecule per row   -> no cross-document context
    V2  this script    seq 128, packed stream sliced   -> cross-document context kept
    --  packed 1024    seq 1024, packed stream         -> the current production shape

V1 alone changes two things at once against production (cross-document context *and*
sequence length), so it cannot separate them. V2 holds the sequence length at V1's 128
while keeping the packed stream, which isolates the cross-document variable: a row here
still spans ~3 molecules and their [SEP] boundaries.

Slicing is exact — 1024 = 8 x 128 — so no token is added, dropped or reordered, and a
row keeps the same neighbours it had inside its block.

Usage:
    LEARNING_SOURCE_DIR=<dir> python scripts/make_packed_slices_128.py
    python scripts/make_packed_slices_128.py --src <dataset> --out <dataset>

The source defaults to the BERT training_ready of the configured learning source
(``COMPOUNDS_DATASET_DIR_BERT``) and the output to that path plus ``_slice128``.
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
logger = logging.getLogger("packed_slices")

BLOCK, SLICE = 1024, 128
CHUNK_ROWS = 20000  # source blocks held in memory at once
OUT_SUFFIX = "_slice128"


def write_split(src_path: Path, out_path: Path, split: str) -> int:
    src = load_from_disk(str(src_path))[split]
    out_dir = out_path / split
    out_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = out_dir / "data-00000-of-00001.arrow"

    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    n_rows = 0
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            for start in range(0, len(src), CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, len(src))
                block = np.asarray(src[start:stop]["input_ids"], dtype=np.int32)
                assert block.shape[1] == BLOCK, f"unexpected block length {block.shape[1]}"
                sliced = block.reshape(-1, SLICE)
                values = pa.array(sliced.reshape(-1), type=pa.int32())
                offsets = pa.array(
                    np.arange(0, (sliced.shape[0] + 1) * SLICE, SLICE, dtype=np.int32)
                )
                table = pa.Table.from_arrays(
                    [pa.ListArray.from_arrays(offsets, values)], schema=schema
                )
                writer.write_table(table)
                n_rows += sliced.shape[0]
                logger.info("  %s: %d/%d blocks -> %d rows", split, stop, len(src), n_rows)

    (out_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "citation": "",
                "description": "",
                "features": {
                    "input_ids": {"feature": {"dtype": "int32", "_type": "Value"}, "_type": "Sequence"}
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
                "_fingerprint": f"packed_slice128_{split}",
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
        # paths.py falls back to "" for an unset LEARNING_SOURCE_DIR, which yields a
        # still-absolute "/compounds/..." — check the variable itself, not the result.
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
    parser = ArgumentParser(description="Cut packed 1024-token blocks into 128-token rows")
    parser.add_argument("--src", default=None, help="source dataset (default: COMPOUNDS_DATASET_DIR_BERT)")
    parser.add_argument("--out", default=None, help=f"output dataset (default: <src>{OUT_SUFFIX})")
    parser.add_argument("--splits", default="train,valid", help="comma-separated splits to convert")
    args = parser.parse_args()

    src_path, out_path = resolve_paths(args.src, args.out)
    splits = [s for s in args.splits.split(",") if s]
    logger.info("source: %s", src_path)
    logger.info("output: %s", out_path)

    counts = {split: write_split(src_path, out_path, split) for split in splits}
    (out_path / "dataset_dict.json").write_text(json.dumps({"splits": splits}, indent=2))

    logger.info("verifying by re-reading ...")
    d = load_from_disk(str(out_path))
    for split, expected in counts.items():
        got = len(d[split])
        row = np.asarray(d[split][0]["input_ids"])
        assert got == expected, f"{split}: {got} != {expected}"
        assert len(row) == SLICE, f"{split}: row length {len(row)}"
        logger.info("  %s: %d rows x %d tokens", split, got, len(row))

    first = splits[0]
    src_head = np.asarray(load_from_disk(str(src_path))[first][0]["input_ids"])
    out_head = np.asarray(d[first][0]["input_ids"])
    assert np.array_equal(src_head[:SLICE], out_head), "first slice does not match the source block"
    logger.info("first slice matches the source block — no tokens reordered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
