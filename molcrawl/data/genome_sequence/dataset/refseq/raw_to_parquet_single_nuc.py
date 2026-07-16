"""Convert per-accession .raw files into single-nucleotide parquet for BERT/GPT-2.

Pipeline position
-----------------
This is Phase 3 of the subset/Evo2 species flow:

    Phase 1 (download_by_accession)  ->  extracted_files/<acc>.fna.gz
    Phase 2 (fasta_to_raw_per_acc)   ->  raw_files/<acc>.raw         (pure ACGT)
    Phase 3 (this module)            ->  parquet_bert/<acc>.parquet
                                         parquet_gpt2/<acc>.parquet

Tokenizer
---------
Single-nucleotide vocab (size 10): A=0, T=1, G=2, C=3, N=4, PAD=5, UNK=6,
CLS=7, SEP=8, MASK=9. Raw is already pure ACGT (Phase 2 folds N + IUPAC and
splits), so in practice only ids 0-3 appear in chunk bodies; UNK is reserved
defensively.

Important: this module deliberately does NOT bake MLM masking into the
parquet. The existing genome BERT trainer uses ``DataCollatorForLanguageModeling``
to mask dynamically at train time; pre-baking masks would (a) freeze the same
mask across all epochs (reducing effective data diversity) and (b) double-mask
on top of the collator. We store only ``input_ids`` (+ ``attention_mask`` for
BERT) so training stays dynamic.

Layout
------
- BERT row: ``input_ids`` = [CLS, 510 nucleotide ids, SEP] = length 512,
  ``attention_mask`` = [1] * 512.
- GPT-2 row: ``input_ids`` = 1024 nucleotide ids (no special tokens, causal LM).

The chunk sizes align with the Phase 2 line cap RAW_LINE_LEN=261,120, so each
raw line yields exactly 512 BERT chunks or 255 GPT-2 chunks with zero
boundary loss.
"""

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

NUC_TO_ID = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
PAD_ID = 5
UNK_ID = 6
CLS_ID = 7
SEP_ID = 8
MASK_ID = 9
VOCAB_SIZE = 10

# Default chunk sizes (so [CLS] + 510 + [SEP] = 512 for BERT max_length=512).
DEFAULT_BERT_CHUNK = 510
DEFAULT_GPT2_CHUNK = 1024

# Pre-built byte translation table for fast tokenization: each input byte ->
# its token id. Unknown bytes map to UNK_ID, though pure-ACGT raw never hits it.
_TRANS_BYTES = bytearray(UNK_ID for _ in range(256))
for _c, _i in NUC_TO_ID.items():
    _TRANS_BYTES[ord(_c)] = _i
TRANS_TABLE = bytes(_TRANS_BYTES)

# pyarrow schemas. Fixed-size list keeps row layout predictable; int32 for ids
# (HF datasets cast cleanly), int8 for the trivially-all-ones attention mask.
_BERT_SCHEMA_CACHE: Dict[int, pa.Schema] = {}
_GPT2_SCHEMA_CACHE: Dict[int, pa.Schema] = {}


# Per-window provenance columns (F2-a): ``accession`` identifies the assembly,
# ``contig_id`` the source FASTA record (chromosome / scaffold). Together they
# let Step 4 hold out whole contigs so a genome's adjacent windows never span
# train and eval. Plain strings in Arrow; ParquetWriter dictionary-encodes them
# on disk by default (each accession parquet has only a handful of distinct
# values), so the stored cost is a small dictionary plus one index per row.
_PROV_FIELDS = [
    ("accession", pa.string()),
    ("contig_id", pa.string()),
]


def _bert_schema(chunk_size: int) -> pa.Schema:
    n = chunk_size + 2  # +CLS +SEP
    if n not in _BERT_SCHEMA_CACHE:
        _BERT_SCHEMA_CACHE[n] = pa.schema(
            [
                ("input_ids", pa.list_(pa.int32(), n)),
                ("attention_mask", pa.list_(pa.int8(), n)),
                *_PROV_FIELDS,
            ]
        )
    return _BERT_SCHEMA_CACHE[n]


def _gpt2_schema(chunk_size: int) -> pa.Schema:
    if chunk_size not in _GPT2_SCHEMA_CACHE:
        _GPT2_SCHEMA_CACHE[chunk_size] = pa.schema(
            [("input_ids", pa.list_(pa.int32(), chunk_size)), *_PROV_FIELDS]
        )
    return _GPT2_SCHEMA_CACHE[chunk_size]


# --------------------------------------------------------------------------- #
# Per-accession conversion
# --------------------------------------------------------------------------- #


def _parquet_paths(base_dir: Path, accession: str, models: Iterable[str]) -> dict:
    return {m: base_dir / f"parquet_{m}" / f"{accession}.parquet" for m in models}


def raw_file_to_parquets(
    raw_path: Union[str, Path],
    base_dir: Union[str, Path],
    models: Tuple[str, ...] = ("bert", "gpt2"),
    bert_chunk_size: int = DEFAULT_BERT_CHUNK,
    gpt2_chunk_size: int = DEFAULT_GPT2_CHUNK,
    batch_rows: int = 10_000,
    force: bool = False,
) -> Tuple[str, dict]:
    """Tokenize one .raw file and write one parquet per requested model.

    Single-pass: the raw is read once and chunks are emitted to all requested
    model writers concurrently. Writes are streamed (ParquetWriter + batched
    flushes) to bound memory regardless of accession size.

    Returns (accession, counts) where counts[model] is the number of chunk rows
    written for that model (or -1 if skipped because output already existed).
    """
    raw_path = Path(raw_path)
    base_dir = Path(base_dir)
    accession = raw_path.stem
    out_paths = _parquet_paths(base_dir, accession, models)

    if not force and all(p.exists() and p.stat().st_size > 0 for p in out_paths.values()):
        return accession, {m: -1 for m in models}

    for p in out_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    schemas = {}
    if "bert" in models:
        schemas["bert"] = _bert_schema(bert_chunk_size)
    if "gpt2" in models:
        schemas["gpt2"] = _gpt2_schema(gpt2_chunk_size)

    tmp_paths = {m: p.with_suffix(".parquet.part") for m, p in out_paths.items()}
    writers: Dict[str, pq.ParquetWriter] = {}
    batches: Dict[str, list] = {m: [] for m in models}
    counts: Dict[str, int] = {m: 0 for m in models}

    def _flush(m: str) -> None:
        if not batches[m]:
            return
        writers[m].write_table(pa.Table.from_pylist(batches[m], schema=schemas[m]))
        counts[m] += len(batches[m])
        batches[m] = []

    try:
        for m in models:
            writers[m] = pq.ParquetWriter(str(tmp_paths[m]), schemas[m], compression="snappy")

        bert_n = bert_chunk_size
        gpt2_n = gpt2_chunk_size
        bert_attn = [1] * (bert_n + 2)  # constant across all BERT rows

        with open(raw_path, "rb") as fh:
            for line in fh:
                # Each raw line is ``<contig_id>\t<sequence>`` (F2-a). Split off
                # the contig id, then translate the sequence bytes to token-id
                # bytes in C (one pass). Lines with no tab are treated as the
                # legacy format (contig id unknown) for backward compatibility.
                raw = line.rstrip(b"\n")
                contig_b, sep, seq_b = raw.partition(b"\t")
                if sep:
                    contig_id = contig_b.decode("ascii", "replace")
                    ids = seq_b.translate(TRANS_TABLE)
                else:
                    contig_id = ""
                    ids = raw.translate(TRANS_TABLE)
                L = len(ids)

                if "bert" in models and L >= bert_n:
                    for i in range(0, L - bert_n + 1, bert_n):
                        chunk = list(ids[i : i + bert_n])
                        batches["bert"].append(
                            {
                                "input_ids": [CLS_ID] + chunk + [SEP_ID],
                                "attention_mask": bert_attn,
                                "accession": accession,
                                "contig_id": contig_id,
                            }
                        )
                        if len(batches["bert"]) >= batch_rows:
                            _flush("bert")

                if "gpt2" in models and L >= gpt2_n:
                    for i in range(0, L - gpt2_n + 1, gpt2_n):
                        batches["gpt2"].append(
                            {
                                "input_ids": list(ids[i : i + gpt2_n]),
                                "accession": accession,
                                "contig_id": contig_id,
                            }
                        )
                        if len(batches["gpt2"]) >= batch_rows:
                            _flush("gpt2")

        for m in models:
            _flush(m)

        # close() writes the footer / page-index metadata; if it fails the file
        # is structurally invalid (thrift "page header" errors on read). The
        # previous version swallowed close() exceptions and still promoted the
        # .part file, which silently corrupted ~33% of the mammal_centered
        # parquets under Track A / mammal Phase 3 CPU+memory contention. Now
        # we propagate close() errors and (below) validate before promotion.
        for m in models:
            writers[m].close()
        writers.clear()

        # Post-write thrift validation: re-open and iterate every page. This
        # catches any case where close() returned success but the footer /
        # page index is unreadable.
        for m in models:
            try:
                _validate_parquet(tmp_paths[m])
            except Exception as e:
                raise RuntimeError(
                    f"post-write validation failed for {m}={tmp_paths[m]}: {e}"
                ) from e
    except BaseException:
        # Any failure: close stragglers (errors here are secondary) and remove
        # all .part files so the next run regenerates this accession cleanly.
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass
        for p in tmp_paths.values():
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        raise

    # Only promote .part → final once every model's file is validated.
    for m in models:
        os.replace(tmp_paths[m], out_paths[m])

    return accession, counts


def _validate_parquet(path: Path) -> None:
    """Read every batch of ``path`` to surface thrift / page-header corruption.

    Cheap-but-sufficient validator: a thrift footer that points at unreadable
    pages will raise during ``iter_batches`` long before reaching the end.
    Used as a post-write integrity check before promoting .part to final.
    """
    pf = pq.ParquetFile(str(path))
    for _ in pf.iter_batches(batch_size=100_000):
        pass


def raw_to_parquet_per_accession(
    base_dir: Union[str, Path],
    models: Tuple[str, ...] = ("bert", "gpt2"),
    bert_chunk_size: int = DEFAULT_BERT_CHUNK,
    gpt2_chunk_size: int = DEFAULT_GPT2_CHUNK,
    num_worker: int = 8,
    batch_rows: int = 10_000,
    force: bool = False,
) -> bool:
    """Convert every ``raw_files/*.raw`` under ``base_dir`` to per-accession parquets.

    Output:
        ``base_dir/parquet_bert/<accession>.parquet``
        ``base_dir/parquet_gpt2/<accession>.parquet``

    Skips accessions whose target parquets already exist (per model). On a
    fully-successful run, touches ``raw_to_parquet_complete.marker``.
    """
    base_dir = Path(base_dir)
    raw_dir = base_dir / "raw_files"
    raw_files = sorted(raw_dir.glob("*.raw"))
    if not raw_files:
        logger.error(f"No .raw files found in {raw_dir}")
        return False

    logger.info(
        f"raw → parquet (per-accession, models={list(models)}, "
        f"bert={bert_chunk_size}, gpt2={gpt2_chunk_size}, workers={num_worker}): "
        f"{len(raw_files)} accessions"
    )

    n_done = n_skip = 0
    total_rows = {m: 0 for m in models}

    def _one(rp: Path):
        return raw_file_to_parquets(
            rp, base_dir, models, bert_chunk_size, gpt2_chunk_size, batch_rows, force
        )

    with concurrent.futures.ThreadPoolExecutor(num_worker) as ex:
        for _acc, counts in ex.map(_one, raw_files):
            if all(v == -1 for v in counts.values()):
                n_skip += 1
            else:
                n_done += 1
                for m, v in counts.items():
                    if v >= 0:
                        total_rows[m] += v

    parquet_counts = {m: len(list((base_dir / f"parquet_{m}").glob("*.parquet"))) for m in models}
    rows_str = ", ".join(f"{m}={total_rows[m]:,}" for m in models)
    files_str = ", ".join(f"{m}={parquet_counts[m]}" for m in models)
    logger.info(
        f"raw → parquet done: converted={n_done} skipped={n_skip} "
        f"files({files_str}) rows({rows_str})"
    )

    if any(parquet_counts[m] == 0 for m in models):
        return False
    (base_dir / "raw_to_parquet_complete.marker").touch()
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert per-accession .raw to single-nucleotide parquet for BERT/GPT-2.",
    )
    p.add_argument("--base", required=True, help="Subset base dir containing raw_files/.")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["bert", "gpt2"],
        default=["bert", "gpt2"],
    )
    p.add_argument("--bert-chunk-size", type=int, default=DEFAULT_BERT_CHUNK)
    p.add_argument("--gpt2-chunk-size", type=int, default=DEFAULT_GPT2_CHUNK)
    p.add_argument("--num-worker", type=int, default=8)
    p.add_argument("--batch-rows", type=int, default=10_000)
    p.add_argument("--force", action="store_true")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    ok = raw_to_parquet_per_accession(
        base_dir=args.base,
        models=tuple(args.models),
        bert_chunk_size=args.bert_chunk_size,
        gpt2_chunk_size=args.gpt2_chunk_size,
        num_worker=args.num_worker,
        batch_rows=args.batch_rows,
        force=args.force,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
