"""Phase 1-2 v3: regenerate compounds training_ready with seq_len=128 filter.

v3 refactor: write Arrow files directly via pyarrow instead of relying on
HuggingFace datasets.save_to_disk (v2 died silently during that call).
Also saves BERT immediately after building, before GPT-2, so partial
progress survives.

Output paths (new dir; existing 3月版 symlink at 20260316/20260520 untouched):
  learning_source_20260708_compounds/compounds/organix13/training_ready_hf_dataset_bert/
    dataset_dict.json + {train,valid,test}/ each with data-00000-of-00001.arrow
  learning_source_20260708_compounds/compounds/organix13/training_ready_hf_dataset_gpt2/
    same shape, input_ids only (no attention_mask).

Reports at the end:
  (a) max token length in each variant (must be = 128)
  (b) truncate count = 0 (filter, not truncate)
  (c) train/valid/test row counts
  (d) actual token count (pad excluded)
"""
from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase1_2_v3")

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
sys.path.insert(0, str(REPO))

BUILD = Path("/lustre/home/matsubara/learning_source_20260716_compounds_v3")
INPUT_PARQUET = BUILD / "compounds" / "organix13" / "OrganiX13.parquet"
OUT_BERT = BUILD / "compounds" / "organix13" / "training_ready_hf_dataset_bert"
OUT_GPT2 = BUILD / "compounds" / "organix13" / "training_ready_hf_dataset_gpt2"

SEQ_LEN = 128
PAD_ID = 0
CLS_ID = 12
SEP_ID = 13
BERT_REAL_CAP = SEQ_LEN - 2   # 126
GPT2_REAL_CAP = SEQ_LEN       # 128
CHUNK = 5000

_tokenizer = None


def _init_worker():
    global _tokenizer
    from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer
    _tokenizer = CompoundsTokenizer(str(REPO / "assets/molecules/vocab.txt"), max_len=1_000_000)


def _tokenize_batch(args):
    global _tokenizer
    smi_list, base_idx = args
    out = []
    for i, s in enumerate(smi_list):
        try:
            ids = _tokenizer.encode(s, padding=False, truncation=False, add_special_tokens=False)
        except Exception:
            ids = []
        out.append((base_idx + i, ids))
    return out


def _write_hf_arrow(out_dir: Path, arrays: dict[str, np.ndarray], split_name: str) -> None:
    """Write a single split as HuggingFace-compatible Arrow files.

    Expected layout:
      out_dir/split_name/data-00000-of-00001.arrow
      out_dir/split_name/dataset_info.json
      out_dir/split_name/state.json
    """
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Build a pyarrow Table with each 2D numpy row as a list<int32> Arrow value.
    # Columns are: "input_ids" (list<int32>) and optionally "attention_mask".
    fields = []
    columns = {}
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr, dtype=np.int32)
        n_rows, seq_len = arr.shape
        # Build a FixedSizeList array in Arrow — much cheaper than list<int32>.
        # But HF datasets expects list<int32> for input_ids. Use ListArray from
        # offsets to keep zero-copy on the underlying int32 buffer.
        values = pa.array(arr.reshape(-1), type=pa.int32())
        offsets = pa.array(np.arange(0, (n_rows + 1) * seq_len, seq_len, dtype=np.int32))
        lst = pa.ListArray.from_arrays(offsets, values)
        columns[name] = lst
        fields.append(pa.field(name, pa.list_(pa.int32())))
    schema = pa.schema(fields)
    table = pa.Table.from_arrays(list(columns.values()), schema=schema)

    arrow_path = split_dir / "data-00000-of-00001.arrow"
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    n_rows = table.num_rows
    # HF-compatible sidecar files so load_from_disk works.
    dataset_info = {
        "citation": "",
        "description": "",
        "features": {
            name: {"feature": {"dtype": "int32", "_type": "Value"}, "_type": "Sequence"}
            for name in arrays.keys()
        },
        "homepage": "",
        "license": "",
        "splits": {
            split_name: {
                "name": split_name,
                "num_bytes": int(arrow_path.stat().st_size),
                "num_examples": int(n_rows),
                "dataset_name": None,
            }
        },
    }
    (split_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))
    state = {
        "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
        "_fingerprint": f"phase1_2_{split_name}",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": split_name,
    }
    (split_dir / "state.json").write_text(json.dumps(state, indent=2))
    logger.info("  wrote %s (%d rows, %.1f MB)", arrow_path,
                n_rows, arrow_path.stat().st_size / 1024**2)


def _write_dataset_dict(out_dir: Path) -> None:
    (out_dir / "dataset_dict.json").write_text(json.dumps(
        {"splits": ["train", "valid", "test"]}, indent=2))


def build_and_save(smi_arr):
    n = len(smi_arr)
    logger.info("[step 1/5] tokenize %d SMILES in parallel (16 workers)...", n)
    t0 = time.time()

    raw_lens = np.zeros(n, dtype=np.int32)
    all_ids_flat_parts = []

    with mp.Pool(16, initializer=_init_worker) as pool:
        args = [(smi_arr[i:i+CHUNK], i) for i in range(0, n, CHUNK)]
        done = 0
        for chunk_out in pool.imap(_tokenize_batch, args, chunksize=1):
            chunk_out.sort(key=lambda x: x[0])
            for idx, ids in chunk_out:
                raw_lens[idx] = len(ids)
            flat = np.concatenate([
                np.asarray(ids, dtype=np.int32) for _, ids in chunk_out
            ]) if chunk_out else np.zeros(0, dtype=np.int32)
            all_ids_flat_parts.append(flat)
            done += len(chunk_out)
            if done % 1_000_000 == 0 or done == n:
                logger.info("    tokenized %d/%d in %.1fs", done, n, time.time()-t0)

    all_ids_flat = np.concatenate(all_ids_flat_parts).astype(np.int16)
    del all_ids_flat_parts
    row_end = np.cumsum(raw_lens, dtype=np.int64)
    row_start = np.concatenate([[0], row_end[:-1]])
    logger.info("  total flat tokens: %d (memory %.2f GB)",
                len(all_ids_flat), all_ids_flat.nbytes / 1024**3)

    bert_mask = (raw_lens >= 1) & (raw_lens <= BERT_REAL_CAP)
    gpt2_mask = (raw_lens >= 1) & (raw_lens <= GPT2_REAL_CAP)

    logger.info("[step 2/5] filter counts")
    logger.info("  raw SMILES:                        %d", n)
    logger.info("  ≥1 token (valid parse):            %d (%d rejected)",
                int((raw_lens >= 1).sum()), int((raw_lens < 1).sum()))
    logger.info("  BERT (real ≤ %d):                 %d (%d dropped as >%d)",
                BERT_REAL_CAP, int(bert_mask.sum()),
                int((raw_lens > BERT_REAL_CAP).sum()), BERT_REAL_CAP)
    logger.info("  GPT-2 (real ≤ %d):                %d (%d dropped as >%d)",
                GPT2_REAL_CAP, int(gpt2_mask.sum()),
                int((raw_lens > GPT2_REAL_CAP).sum()), GPT2_REAL_CAP)

    rng = np.random.default_rng(42)
    split_assign = rng.integers(0, 100, size=n, dtype=np.int32)
    split_ids = np.where(split_assign < 80, 0,
                np.where(split_assign < 90, 1, 2)).astype(np.int8)

    # ============ BERT ============
    OUT_BERT.mkdir(parents=True, exist_ok=True)
    logger.info("[step 3/5] BERT: build + save per split")
    for split_code, split_name in [(0, "train"), (1, "valid"), (2, "test")]:
        split_mask = bert_mask & (split_ids == split_code)
        idx = np.where(split_mask)[0]
        n_rows = len(idx)
        logger.info("  BERT/%s: building %d rows ...", split_name, n_rows)
        t0 = time.time()
        input_ids = np.full((n_rows, SEQ_LEN), PAD_ID, dtype=np.int16)
        attn = np.zeros((n_rows, SEQ_LEN), dtype=np.int8)
        for out_row, i in enumerate(idx):
            rl = int(raw_lens[i])
            s, e = row_start[i], row_end[i]
            input_ids[out_row, 0] = CLS_ID
            input_ids[out_row, 1:1+rl] = all_ids_flat[s:e]
            input_ids[out_row, 1+rl] = SEP_ID
            attn[out_row, :2+rl] = 1
        logger.info("    built in %.1fs (memory arr=%.2f GB)",
                    time.time()-t0,
                    (input_ids.nbytes + attn.nbytes) / 1024**3)
        _write_hf_arrow(OUT_BERT, {"input_ids": input_ids, "attention_mask": attn}, split_name)
        del input_ids, attn
        gc.collect()
    _write_dataset_dict(OUT_BERT)
    logger.info("  BERT DatasetDict written to %s", OUT_BERT)

    # ============ GPT-2 ============
    OUT_GPT2.mkdir(parents=True, exist_ok=True)
    logger.info("[step 4/5] GPT-2: build + save per split")
    for split_code, split_name in [(0, "train"), (1, "valid"), (2, "test")]:
        split_mask = gpt2_mask & (split_ids == split_code)
        idx = np.where(split_mask)[0]
        n_rows = len(idx)
        logger.info("  GPT-2/%s: building %d rows ...", split_name, n_rows)
        t0 = time.time()
        input_ids = np.full((n_rows, SEQ_LEN), PAD_ID, dtype=np.int16)
        for out_row, i in enumerate(idx):
            rl = int(raw_lens[i])
            s, e = row_start[i], row_end[i]
            input_ids[out_row, :rl] = all_ids_flat[s:e]
        logger.info("    built in %.1fs (memory %.2f GB)",
                    time.time()-t0, input_ids.nbytes / 1024**3)
        _write_hf_arrow(OUT_GPT2, {"input_ids": input_ids}, split_name)
        del input_ids
        gc.collect()
    _write_dataset_dict(OUT_GPT2)
    logger.info("  GPT-2 DatasetDict written to %s", OUT_GPT2)

    del all_ids_flat, row_start, row_end, raw_lens
    gc.collect()


def verify():
    logger.info("[step 5/5] verify by re-reading via HF datasets ...")
    from datasets import load_from_disk
    for label, out_dir in [("BERT", OUT_BERT), ("GPT-2", OUT_GPT2)]:
        logger.info("=== verify %s at %s ===", label, out_dir)
        d = load_from_disk(str(out_dir))
        total_rows = 0
        total_pad_incl = 0
        total_pad_excl = 0
        max_seq_len = 0
        for sp in ("train", "valid", "test"):
            s = d[sp]
            n = len(s)
            first_len = len(s[0]["input_ids"])
            assert first_len == SEQ_LEN, f"{label}/{sp} row 0 length {first_len} != {SEQ_LEN}"
            max_seq_len = max(max_seq_len, first_len)
            excl = 0
            batch = 10000
            for st in range(0, n, batch):
                end = min(st + batch, n)
                a = np.asarray(s[st:end]["input_ids"], dtype=np.int64)
                excl += int((a != PAD_ID).sum())
            total_rows += n
            total_pad_incl += n * SEQ_LEN
            total_pad_excl += excl
            logger.info("  %s/%s: rows=%d  seq_len=%d  pad_incl=%d  pad_excl=%d (%.1f%%)",
                        label, sp, n, SEQ_LEN, n * SEQ_LEN, excl, 100 * excl / (n * SEQ_LEN))
        logger.info("  [%s totals] rows=%d  pad_incl=%d  pad_excl=%d  max_seq_len=%d",
                    label, total_rows, total_pad_incl, total_pad_excl, max_seq_len)


def main() -> int:
    logger.info("=== Phase 1-2 v3 (pyarrow direct write): seq_len=%d filter ===", SEQ_LEN)
    logger.info("input:  %s", INPUT_PARQUET)
    logger.info("out BERT: %s", OUT_BERT)
    logger.info("out GPT-2: %s", OUT_GPT2)

    tbl = pq.read_table(str(INPUT_PARQUET), columns=["smiles"])
    smi_arr = tbl.column("smiles").to_pylist()
    logger.info("loaded %d SMILES", len(smi_arr))

    build_and_save(smi_arr)
    verify()
    logger.info("=== Phase 1-2 completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
