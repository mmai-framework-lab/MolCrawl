"""Phase 1-2 v4: regenerate compounds training_ready as packed 1024-token blocks.

Supersedes ``run_phase1_2_training_ready_v3.py`` (1 molecule = 1 row, seq_len=128,
pad 70.7%).  v4 packs the SMILES token stream into fixed 1024-token blocks the way
protein / genome / rna / molecule_nat_lang already do, so every modality shares the
same pretraining input shape:

    [mol_0] [SEP] [mol_1] [SEP] ... -> chunk into 1024 -> drop the trailing remainder

Differences from v3 that matter for the research numbers:
  * No length filter.  v3 dropped 58,273 molecules (0.449%) whose token length
    exceeded the 128 cap; packing has no per-molecule cap, so they come back.
  * No padding at all, hence no pad-loss masking is needed downstream.
  * Row count drops ~26x (12.93M rows of 128 -> ~0.40M blocks of 1024), so
    ``max_iters`` / ``warmup`` / ``eval_interval`` must be recomputed from the
    row counts this script reports.

GPT-2 and BERT read byte-identical data (``input_ids`` only, matching the protein
training_ready layout), and with ``PREP_OUT_SUFFIX`` unset both directories are written
where the existing config paths (``COMPOUNDS_DATASET_DIR_GPT2`` / ``_BERT``) already
point, so those configs keep working unchanged.

Split assignment reuses v3's ``default_rng(42)`` draw over the input parquet order,
so a molecule present in both builds lands in the same split.  Packing is done
*after* splitting, so no block ever straddles a split boundary.  Within a split the
molecules are permuted before packing (``PACK_ORDER_SEED``) so a block is not a run
of consecutive parquet rows; see the constant for why that matters.

Required env:
  MOLCRAWL_REPO  -> repo root (for assets/molecules/vocab.txt).  Default: this repo.
  ORGANIX13_DIR  -> {LEARNING_SOURCE_DIR}/compounds/organix13 (input + output root).
Optional env:
  V4_LIMIT       -> only process the first N SMILES (dry run / validation).
  PACK_ORDER     -> "shuffled" (default) or "source". "source" reproduces the
                    2026-08-05 v4 build, which concatenated in parquet order.
  PREP_OUT_SUFFIX-> appended to both output dir names, and read by the prep modules
                    for the other modalities too. Set it for any rebuild that is not
                    meant to replace the data the finished runs used, e.g.
                    PREP_OUT_SUFFIX=_shuffled -> training_ready_hf_dataset_gpt2_shuffled.
  V4_WORKERS     -> tokenizer worker processes (default 16).
"""
from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase1_2_v4")

REPO = Path(os.environ.get("MOLCRAWL_REPO", "/data1/rkp00024/matsubara/MolCrawl"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_ORG = Path(os.environ["ORGANIX13_DIR"])
INPUT_PARQUET = _ORG / "OrganiX13.parquet"
# Output suffix, shared with the prep modules under molcrawl/data so that one
# variable covers a rebuild of every modality. A second name here would mean a
# five-modality rebuild that sets only the shared one silently overwrites compounds
# while the other four are safely diverted — the worst arrangement, because the
# four that worked make the fifth look safe too.
OUT_SUFFIX = os.environ.get("PREP_OUT_SUFFIX", "")
OUT_BERT = _ORG / f"training_ready_hf_dataset_bert{OUT_SUFFIX}"
OUT_GPT2 = _ORG / f"training_ready_hf_dataset_gpt2{OUT_SUFFIX}"

CONTEXT_LENGTH = 1024
PAD_ID = 0            # never appears in v4 output; kept for the verify assertion
SEP_ID = 13           # CompoundsTokenizer.eos_token_id — the molecule separator
SPLIT_SEED = 42       # same draw as v3 so split membership is stable
# Order the molecules are concatenated in before the stream is cut into blocks.
# The parquet is not in random order (the first million rows top out at 68 tokens
# against 128 for the whole file), so packing it as-is puts ~26 mutually similar
# molecules in every block and keeps them there for the whole run.
#
# The cost is block homogeneity, not batch diversity: get_batch redraws the rows
# every iteration, so a batch is still a random sample of blocks. What is fixed is
# which molecules share a block — and GPT-2 attends causally over the whole block
# with no document mask (models/gpt2/model.py: attn_mask=None, is_causal=True), so
# in a homogeneous block a molecule can look back at near-copies of itself.
#
# Permuting the split first removes that correlation; set PACK_ORDER=source to
# reproduce the 2026-08-05 v4 build byte for byte.
PACK_ORDER = os.environ.get("PACK_ORDER", "shuffled")
if PACK_ORDER not in ("shuffled", "source"):
    # Checked here rather than at the packing step: a typo would otherwise surface
    # only after tokenising 13M SMILES across 16 processes.
    raise SystemExit(f"PACK_ORDER must be 'shuffled' or 'source', got {PACK_ORDER!r}")
PACK_ORDER_SEED = 43
CHUNK = 5000

# Ladder settings the configs must match (see tmp/compounds-reply-to-decision-2026-08-05.md).
EPOCHS = 10
GLOBAL_BATCH_SEQ = 2560

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


def _write_hf_arrow(out_dir: Path, blocks: np.ndarray, split_name: str) -> None:
    """Write one split as HuggingFace-compatible Arrow (input_ids only)."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    arr = np.ascontiguousarray(blocks, dtype=np.int32)
    n_rows, seq_len = arr.shape
    values = pa.array(arr.reshape(-1), type=pa.int32())
    offsets = pa.array(np.arange(0, (n_rows + 1) * seq_len, seq_len, dtype=np.int32))
    lst = pa.ListArray.from_arrays(offsets, values)
    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    table = pa.Table.from_arrays([lst], schema=schema)

    arrow_path = split_dir / "data-00000-of-00001.arrow"
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    dataset_info = {
        "citation": "",
        "description": "",
        "features": {
            "input_ids": {"feature": {"dtype": "int32", "_type": "Value"}, "_type": "Sequence"}
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
        "_fingerprint": f"phase1_2_v4_{split_name}",
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": split_name,
    }
    (split_dir / "state.json").write_text(json.dumps(state, indent=2))
    logger.info(
        "  wrote %s (%d blocks, %.1f MB)", arrow_path, n_rows, arrow_path.stat().st_size / 1024**2
    )


def _write_dataset_dict(out_dir: Path) -> None:
    (out_dir / "dataset_dict.json").write_text(
        json.dumps({"splits": ["train", "valid", "test"]}, indent=2)
    )


def _split_order(idx, split_code):
    """Order one split's molecules for packing.

    Permutes within the split only: membership stays exactly as the SPLIT_SEED draw
    assigned it, so this changes which molecules share a block, never which split a
    molecule is in. A per-split offset keeps the three splits from sharing a
    permutation. PACK_ORDER=source returns the parquet order untouched.

    The permuted order reads ``all_ids_flat`` randomly instead of sequentially, which
    costs cache misses across a multi-GB array; the Python loop in _pack_split
    dominates, but a rebuild may still take longer than the source-order one did.
    """
    if PACK_ORDER == "source":
        return idx
    return np.random.default_rng(PACK_ORDER_SEED + split_code).permutation(idx)


def _pack_split(idx, raw_lens, row_start, row_end, all_ids_flat):
    """Concatenate the split's molecules with [SEP] and cut into 1024-token blocks."""
    n_mol = len(idx)
    stream_len = int(raw_lens[idx].sum()) + n_mol  # +1 [SEP] per molecule
    stream = np.empty(stream_len, dtype=np.int16)
    pos = 0
    for i in idx:
        rl = int(raw_lens[i])
        if rl:
            stream[pos : pos + rl] = all_ids_flat[row_start[i] : row_end[i]]
            pos += rl
        stream[pos] = SEP_ID
        pos += 1
    assert pos == stream_len, f"stream fill mismatch {pos} != {stream_len}"

    n_blocks = stream_len // CONTEXT_LENGTH
    remainder = stream_len - n_blocks * CONTEXT_LENGTH
    blocks = stream[: n_blocks * CONTEXT_LENGTH].reshape(n_blocks, CONTEXT_LENGTH)
    return blocks, stream_len, remainder


def build_and_save(smi_arr):
    n = len(smi_arr)
    workers = int(os.environ.get("V4_WORKERS", "16"))
    logger.info("[step 1/4] tokenize %d SMILES in parallel (%d workers)...", n, workers)
    t0 = time.time()

    raw_lens = np.zeros(n, dtype=np.int32)
    all_ids_flat_parts = []

    with mp.Pool(workers, initializer=_init_worker) as pool:
        args = [(smi_arr[i : i + CHUNK], i) for i in range(0, n, CHUNK)]
        done = 0
        for chunk_out in pool.imap(_tokenize_batch, args, chunksize=1):
            chunk_out.sort(key=lambda x: x[0])
            for idx, ids in chunk_out:
                raw_lens[idx] = len(ids)
            flat = (
                np.concatenate([np.asarray(ids, dtype=np.int32) for _, ids in chunk_out])
                if chunk_out
                else np.zeros(0, dtype=np.int32)
            )
            all_ids_flat_parts.append(flat)
            done += len(chunk_out)
            if done % 1_000_000 == 0 or done == n:
                logger.info("    tokenized %d/%d in %.1fs", done, n, time.time() - t0)

    all_ids_flat = np.concatenate(all_ids_flat_parts).astype(np.int16)
    del all_ids_flat_parts
    row_end = np.cumsum(raw_lens, dtype=np.int64)
    row_start = np.concatenate([[0], row_end[:-1]])
    logger.info(
        "  total flat tokens: %d (memory %.2f GB)", len(all_ids_flat), all_ids_flat.nbytes / 1024**3
    )

    logger.info("[step 2/4] length stats (no filter, no truncation in v4)")
    logger.info("  raw SMILES:              %d", n)
    logger.info("  >=1 token (valid parse): %d (%d rejected)", int((raw_lens >= 1).sum()), int((raw_lens < 1).sum()))
    logger.info(
        "  real length: mean %.2f  p99 %d  max %d", raw_lens.mean(), int(np.percentile(raw_lens, 99)), int(raw_lens.max())
    )
    logger.info(
        "  molecules v3 would have dropped (>128): %d (%.3f%%) — kept in v4",
        int((raw_lens > 128).sum()),
        100 * float((raw_lens > 128).mean()),
    )

    # Same split draw as v3 (seed 42 over the input parquet order).
    rng = np.random.default_rng(SPLIT_SEED)
    split_assign = rng.integers(0, 100, size=n, dtype=np.int32)
    split_ids = np.where(split_assign < 80, 0, np.where(split_assign < 90, 1, 2)).astype(np.int8)
    valid_mask = raw_lens >= 1

    logger.info("[step 3/4] pack into %d-token blocks and save (per split)", CONTEXT_LENGTH)
    summary = {}
    logger.info("  packing order: %s (seed %d)", PACK_ORDER, PACK_ORDER_SEED)
    for split_code, split_name in [(0, "train"), (1, "valid"), (2, "test")]:
        idx = _split_order(np.where(valid_mask & (split_ids == split_code))[0], split_code)
        t0 = time.time()
        blocks, stream_len, remainder = _pack_split(idx, raw_lens, row_start, row_end, all_ids_flat)
        logger.info(
            "  %s: %d molecules -> %d tokens (incl [SEP]) -> %d blocks (dropped remainder %d tokens) in %.1fs",
            split_name,
            len(idx),
            stream_len,
            len(blocks),
            remainder,
            time.time() - t0,
        )
        for out_dir in (OUT_GPT2, OUT_BERT):
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_hf_arrow(out_dir, blocks, split_name)
        summary[split_name] = {
            "molecules": int(len(idx)),
            "tokens": int(stream_len),
            "blocks": int(len(blocks)),
            "dropped_remainder_tokens": int(remainder),
        }
        del blocks
        gc.collect()

    for out_dir in (OUT_GPT2, OUT_BERT):
        _write_dataset_dict(out_dir)
        logger.info("  DatasetDict written to %s", out_dir)

    del all_ids_flat, row_start, row_end, raw_lens
    gc.collect()
    return summary


def verify(summary):
    logger.info("[step 4/4] verify by re-reading via HF datasets ...")
    from datasets import load_from_disk

    for label, out_dir in [("GPT-2", OUT_GPT2), ("BERT", OUT_BERT)]:
        logger.info("=== verify %s at %s ===", label, out_dir)
        d = load_from_disk(str(out_dir))
        for sp in ("train", "valid", "test"):
            s = d[sp]
            n = len(s)
            assert n == summary[sp]["blocks"], f"{label}/{sp}: {n} != {summary[sp]['blocks']}"
            probe = min(n, 20000)
            a = np.asarray(s[0:probe]["input_ids"], dtype=np.int64)
            assert a.shape[1] == CONTEXT_LENGTH, f"{label}/{sp} block length {a.shape[1]}"
            n_pad = int((a == PAD_ID).sum())
            n_sep = int((a == SEP_ID).sum())
            logger.info(
                "  %s/%s: blocks=%d  block_len=%d  probe=%d rows: pad=%d (must be 0)  [SEP]=%d (%.1f/block)",
                label,
                sp,
                n,
                a.shape[1],
                probe,
                n_pad,
                n_sep,
                n_sep / probe,
            )
            assert n_pad == 0, f"{label}/{sp}: packed data must not contain PAD"

    train_blocks = summary["train"]["blocks"]
    max_iters = train_blocks * EPOCHS // GLOBAL_BATCH_SEQ
    logger.info("=== config numbers derived from this build ===")
    logger.info("  train blocks:                 %d", train_blocks)
    logger.info("  epochs / global batch (seq):  %d / %d", EPOCHS, GLOBAL_BATCH_SEQ)
    logger.info("  max_iters (= max_steps):      %d", max_iters)
    logger.info("  warmup (2%% of max_iters):     %d", max(1, round(max_iters * 0.02)))
    logger.info("  tokens per iteration:         %d", GLOBAL_BATCH_SEQ * CONTEXT_LENGTH)
    logger.info("  tokens processed (%d epochs): %d", EPOCHS, max_iters * GLOBAL_BATCH_SEQ * CONTEXT_LENGTH)


def main() -> int:
    logger.info("=== Phase 1-2 v4 (packed %d, no filter, no truncation) ===", CONTEXT_LENGTH)
    logger.info("input:     %s", INPUT_PARQUET)
    logger.info("out GPT-2: %s", OUT_GPT2)
    logger.info("out BERT:  %s", OUT_BERT)

    tbl = pq.read_table(str(INPUT_PARQUET), columns=["smiles"])
    smi_arr = tbl.column("smiles").to_pylist()
    limit = os.environ.get("V4_LIMIT")
    if limit:
        smi_arr = smi_arr[: int(limit)]
        logger.warning("V4_LIMIT set — DRY RUN over the first %d SMILES only", len(smi_arr))
    logger.info("loaded %d SMILES", len(smi_arr))

    summary = build_and_save(smi_arr)
    verify(summary)
    logger.info("=== Phase 1-2 v4 completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
