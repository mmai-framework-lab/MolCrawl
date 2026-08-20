"""A/B probe: one protein per 1024 block, unpacked from the packed protein set.

The packed protein dataset concatenates ~3-5 proteins into each 1024 block,
separated by <eos> (id 2). Document masking (confine attention per protein) did not
break the collapse (job 24870). This script drops the packing entirely: it splits
each packed block on <eos> and re-emits each protein one-per-row, padded to 1024,
with an attention_mask so the padding is not attended to.

If BERT learns on this (1 protein / block, no packing, document_masking off), the
culprit is the packing / document count / masking path. If it still collapses, the
problem is deeper (LR / data) and document masking cannot fix it. Same tokenizer and
same tokens as the packed set — only the packing changes.

Note: at one protein per 1024 the padding is heavy (proteins avg ~250-350 aa), so a
step processes far fewer real tokens than the packed set. This is a diagnostic, not a
production build.
"""
from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("single_protein_1024")

TARGET_LEN = 1024
PAD_ID = 1      # protein <pad>
SEP_ID = 2      # protein <eos> — the inter-document separator in the packed data
READ_CHUNK = 20000   # blocks read per batch
WRITE_CHUNK = 50000  # proteins per arrow write


def _write_arrays(writer, buf, schema):
    m = len(buf)
    out_ids = np.full((m, TARGET_LEN), PAD_ID, dtype=np.int32)
    out_mask = np.zeros((m, TARGET_LEN), dtype=np.int32)
    for k, prot in enumerate(buf):
        L = min(len(prot), TARGET_LEN)
        out_ids[k, :L] = prot[:L]
        out_mask[k, :L] = 1
    arrays = []
    for arr in (out_ids, out_mask):
        values = pa.array(arr.reshape(-1), type=pa.int32())
        offsets = pa.array(np.arange(0, (m + 1) * TARGET_LEN, TARGET_LEN, dtype=np.int32))
        arrays.append(pa.ListArray.from_arrays(offsets, values))
    writer.write_table(pa.Table.from_arrays(arrays, schema=schema))


def unpack_split(src_path: Path, out_path: Path, split: str, limit: int) -> int:
    ds = load_from_disk(str(src_path))[split]
    out_dir = out_path / split
    out_dir.mkdir(parents=True, exist_ok=True)
    arrow_path = out_dir / "data-00000-of-00001.arrow"
    schema = pa.schema(
        [pa.field("input_ids", pa.list_(pa.int32())), pa.field("attention_mask", pa.list_(pa.int32()))]
    )
    n = 0
    buf: list[list[int]] = []
    done = False
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_stream(sink, schema) as writer:
            for start in range(0, len(ds), READ_CHUNK):
                stop = min(start + READ_CHUNK, len(ds))
                blocks = ds[start:stop]["input_ids"]
                for ids in blocks:
                    seg: list[int] = []
                    for t in ids:
                        seg.append(int(t))
                        if t == SEP_ID:
                            if len(seg) > 1:  # skip empty/degenerate segments
                                buf.append(seg)
                                n += 1
                            seg = []
                            if len(buf) >= WRITE_CHUNK:
                                _write_arrays(writer, buf, schema)
                                buf = []
                            if n >= limit:
                                done = True
                                break
                    if done:
                        break
                logger.info("  %s: %d proteins (through block %d/%d)", split, n, stop, len(ds))
                if done:
                    break
            if buf:
                _write_arrays(writer, buf, schema)

    (out_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "citation": "",
                "description": "one protein per 1024 block (unpacked A/B probe)",
                "features": {
                    name: {"feature": {"dtype": "int32", "_type": "Value"}, "_type": "Sequence"}
                    for name in ("input_ids", "attention_mask")
                },
                "homepage": "",
                "license": "",
                "splits": {split: {"name": split, "num_bytes": int(arrow_path.stat().st_size), "num_examples": int(n), "dataset_name": None}},
            },
            indent=2,
        )
    )
    (out_dir / "state.json").write_text(
        json.dumps(
            {
                "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
                "_fingerprint": f"single_protein_1024_{split}",
                "_format_columns": None,
                "_format_kwargs": {},
                "_format_type": None,
                "_output_all_columns": False,
                "_split": split,
            },
            indent=2,
        )
    )
    return n


def main() -> int:
    parser = ArgumentParser(description="Unpack packed protein data to one protein per 1024 block")
    parser.add_argument("--src", required=True, help="packed protein dataset (DatasetDict with train/valid)")
    parser.add_argument("--out", required=True, help="output dataset dir")
    parser.add_argument("--train-proteins", type=int, default=1_000_000)
    parser.add_argument("--valid-proteins", type=int, default=50_000)
    args = parser.parse_args()

    src_path, out_path = Path(args.src), Path(args.out)
    logger.info("source: %s", src_path)
    logger.info("output: %s", out_path)
    counts = {
        "train": unpack_split(src_path, out_path, "train", args.train_proteins),
        "valid": unpack_split(src_path, out_path, "valid", args.valid_proteins),
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
        assert int(mask.sum()) == int((ids != PAD_ID).sum()), f"{split}: mask/padding mismatch"
        logger.info(
            "  %s: %d rows x %d tokens, %d real in row0 (%.1f%% padding)",
            split, len(d[split]), len(ids), int(mask.sum()), 100 * (1 - mask.sum() / TARGET_LEN),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
