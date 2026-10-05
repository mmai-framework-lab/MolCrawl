"""Turn fetched ZINC-22 .smi.gz tranche files into packed token shards.

One shard per input file, so the work splits across a Slurm array without any two
tasks writing the same output and a re-run only redoes the files whose shard is
missing. Assembling the shards into a DatasetDict is a separate, cheap step
(zinc22_assemble.py) that has to see all of them.

The packing matches what organix13's training_ready_hf_dataset already holds, so the
two corpora can be trained with the same configs: the existing 612-entry vocabulary
at assets/molecules/vocab.txt, molecules joined by [SEP], cut into fixed blocks of
1,024, written as int32. Nothing is sampled down and no molecule is dropped for
being an ion -- the ZINC-22 sample carries 1.0% charged atoms and no multi-fragment
entries, against 4.3% in organix13, so dropping them would make the new corpus less
like the old one rather than more. The per-file counts record how many there were, so
that choice can be revisited without re-reading the corpus.

A molecule longer than the block size is kept: it spans a block boundary exactly as a
short one does, which is what the organix13 packing does too.

The tranche sizes are heavily skewed -- 1,309 files under a megabyte against 304 over
a gigabyte, the largest 11 GiB -- so one process per file would leave the whole array
waiting on a handful of tasks. Lines are therefore handed to a pool in ordered chunks
and the blocks are cut in the parent from the results in order, which keeps the output
identical to the single-process one regardless of how many workers run.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BLOCK = 1024
REPORT_EVERY = 2_000_000
CHUNK_LINES = 20_000

_TOK = None
_UNK = None
_COLUMN = 0


def _init_worker(vocab_path, block, unk_id, smiles_column):
    """Build one tokenizer per worker: constructing it per chunk would dominate."""
    global _TOK, _UNK, _COLUMN
    from molcrawl.data.compounds.utils.tokenizer import SmilesTokenizer

    _TOK = SmilesTokenizer(vocab_path, model_max_length=block)
    _UNK = unk_id
    _COLUMN = smiles_column


def _encode_chunk(lines):
    """Encode one chunk of .smi lines. Returns ids plus the counts for those lines."""
    out = []
    n_tok = n_unknown = n_charged = n_fragment = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if _COLUMN >= len(parts):
            continue
        smi = parts[_COLUMN]
        if smi.lower() in ("smiles", "smi"):
            continue
        ids = _TOK.encode(smi, add_special_tokens=False)
        if not ids:
            continue
        out.append(ids)
        n_tok += len(ids)
        if _UNK in ids:
            n_unknown += 1
        if "+" in smi or "-" in smi:
            n_charged += 1
        if "." in smi:
            n_fragment += 1
    return out, n_tok, n_unknown, n_charged, n_fragment


def load_vocab(path):
    """Read the WordPiece vocabulary as token -> id, in file order."""
    with open(path, encoding="utf-8") as fh:
        return {line.rstrip("\n"): i for i, line in enumerate(fh)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smi", required=True, help="one .smi.gz tranche file")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True, help="parquet shard to write")
    ap.add_argument("--counts", required=True, help="JSON of what this file contained")
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--smiles-column", type=int, default=0,
                    help="column of the whitespace-separated .smi line holding the SMILES")
    # os.cpu_count() reports the node's cores, not the ones allocated to this task,
    # so the allocation is the default when Slurm states it.
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 2))
    args = ap.parse_args(argv)

    if os.path.exists(args.out) and os.path.exists(args.counts):
        print(f"既存 {args.out} -- 何もしない")
        return 0

    vocab = load_vocab(args.vocab)
    sep_id = vocab["[SEP]"]

    stream: list[int] = []
    blocks: list[np.ndarray] = []
    n_mol = n_tok = n_unknown_mol = n_charged = n_fragment = 0
    unk_id = vocab["[UNK]"]
    t0 = time.time()
    next_report = REPORT_EVERY

    with gzip.open(args.smi, "rt", encoding="utf-8", errors="replace") as fh:
        chunks = iter(lambda: list(itertools.islice(fh, CHUNK_LINES)), [])
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(args.vocab, args.block, unk_id, args.smiles_column)) as pool:
            for encoded, c_tok, c_unk, c_chg, c_frag in pool.imap(_encode_chunk, chunks):
                n_tok += c_tok
                n_unknown_mol += c_unk
                n_charged += c_chg
                n_fragment += c_frag
                for ids in encoded:
                    n_mol += 1
                    stream.extend(ids)
                    stream.append(sep_id)
                while len(stream) >= args.block:
                    blocks.append(np.asarray(stream[: args.block], dtype=np.int32))
                    del stream[: args.block]
                if n_mol >= next_report:
                    el = time.time() - t0
                    print(f"  {n_mol:,} 分子 {n_tok:,} トークン {len(blocks):,} ブロック "
                          f"{n_mol / el:,.0f} 分子/秒", flush=True)
                    next_report += REPORT_EVERY

    # The tail shorter than one block is dropped rather than padded: padding would put
    # tokens in the corpus that no molecule produced, and the shards are assembled
    # together afterwards, so one partial block per file is the whole loss.
    dropped_tail = len(stream)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table = pa.table({"input_ids": pa.array(blocks, type=pa.list_(pa.int32()))})
    pq.write_table(table, args.out, compression="zstd")

    counts = {
        "source": os.path.abspath(args.smi),
        "shard": os.path.abspath(args.out),
        "molecules": n_mol,
        "tokens": n_tok,
        "blocks": len(blocks),
        "block_size": args.block,
        "tokens_per_molecule": round(n_tok / n_mol, 3) if n_mol else 0.0,
        "molecules_with_unknown_token": n_unknown_mol,
        "molecules_with_charge_sign": n_charged,
        "molecules_multi_fragment": n_fragment,
        "dropped_tail_tokens": dropped_tail,
        "seconds": round(time.time() - t0, 1),
        "workers": args.workers,
    }
    with open(args.counts, "w", encoding="utf-8") as fh:
        json.dump(counts, fh, ensure_ascii=False, indent=2)

    print(f"{n_mol:,} 分子 / {n_tok:,} トークン / {len(blocks):,} ブロック "
          f"/ 端数 {dropped_tail} トークン破棄 / {counts['seconds']} 秒")
    print(f"shard {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
