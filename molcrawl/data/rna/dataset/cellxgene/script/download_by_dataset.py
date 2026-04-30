"""
Per-dataset download strategy for CellxGene census data.

Problem with the original download.py approach:
  get_anndata(obs_coords=[5000 arbitrary soma_joinids])
  → These IDs span many datasets/TileDB fragments → expensive random I/O.

This module instead:
  1. Queries census obs metadata once per tissue (no X, fast) → soma_joinid→dataset_id
  2. Groups our needed obs_ids by dataset_id
  3. For each dataset: queries get_anndata in sub-batches of _MAX_CELLS_PER_BATCH
     cells, routes them into chunk buffers, and flushes complete chunks to disk
     immediately to keep peak memory low.

Compatible drop-in replacement for download.download().
"""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_SOCKET_TIMEOUT_SEC = 300       # per socket recv
_QUERY_TIMEOUT_SEC = 1800       # 30 min per sub-batch query
_MAX_RETRY = 3
# Max cells per get_anndata call — keeps each query under ~4 GB peak memory.
# brain has datasets with 60K+ cells × 60K genes sparse → ~1.4 GB per 5K cells.
# 20K cells ≈ 5.5 GB peak but that should fit on h200 (80 GB).
_MAX_CELLS_PER_BATCH = 20_000

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Census helpers
# ---------------------------------------------------------------------------

def _open_census(version: str, try_count: int = 0) -> Any:
    import cellxgene_census
    socket.setdefaulttimeout(_SOCKET_TIMEOUT_SEC)
    try:
        return cellxgene_census.open_soma(census_version=version)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if try_count >= _MAX_RETRY:
            raise
        logger.warning(f"open_soma failed (attempt {try_count + 1}): {exc}")
        time.sleep(10)
        return _open_census(version, try_count + 1)


def _get_anndata_for_dataset(
    version: str,
    dataset_id: str,
    obs_ids: List[int],
    target_gene_ids: Sequence[int],
    try_count: int = 0,
) -> Any:
    """get_anndata scoped to one dataset_id.  All obs_ids come from a single
    TileDB fragment → much faster than cross-dataset random access."""
    import cellxgene_census
    import concurrent.futures as cf

    socket.setdefaulttimeout(_SOCKET_TIMEOUT_SEC)
    census = _open_census(version)
    try:
        def _fetch():
            return cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                obs_coords=obs_ids,
                var_coords=target_gene_ids,
            )

        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_fetch)
            try:
                adata = fut.result(timeout=_QUERY_TIMEOUT_SEC)
            except cf.TimeoutError as err:
                ex.shutdown(wait=False)
                raise TimeoutError(
                    f"get_anndata timed out after {_QUERY_TIMEOUT_SEC}s "
                    f"(dataset {dataset_id[:8]}, {len(obs_ids)} cells)"
                ) from err
    except KeyboardInterrupt:
        census.close()
        raise
    except Exception:
        census.close()
        raise
    else:
        census.close()
    return adata


def _get_obs_metadata(
    version: str, tissue: str, soma_joinids: np.ndarray
) -> pd.DataFrame:
    """Single metadata-only census query: returns (soma_joinid, dataset_id).
    No expression matrix is read, so this is fast."""
    logger.info(
        f"[{tissue}] Querying obs metadata for {len(soma_joinids):,} cells "
        f"(1 census call)…"
    )
    census = _open_census(version)
    try:
        obs_df = (
            census["census_data"]["homo_sapiens"]
            .obs.read(
                value_filter=(
                    f"tissue_general == '{tissue}' and is_primary_data == True"
                ),
                column_names=["soma_joinid", "dataset_id"],
            )
            .concat()
            .to_pandas()
        )
    finally:
        census.close()

    needed = set(soma_joinids.tolist())
    obs_df = obs_df[obs_df["soma_joinid"].isin(needed)].copy()
    logger.info(
        f"[{tissue}] Mapped {len(obs_df):,} cells across "
        f"{obs_df['dataset_id'].nunique()} datasets"
    )
    return obs_df


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def _find_remaining_chunks(
    output_dir: Path, size_workload: int
) -> Dict[str, List[Tuple[int, int, np.ndarray]]]:
    """Returns {tissue: [(start, end, obs_ids_array), ...]} for missing chunks."""
    meta_dir = output_dir / "metadata_preparation_dir"
    dl_dir = output_dir / "download_dir"
    remaining: Dict[str, List[Tuple[int, int, np.ndarray]]] = {}

    for obs_file in sorted(meta_dir.glob("*.obs_id.tsv")):
        tissue = obs_file.stem.split(".")[0]
        ids = np.loadtxt(obs_file, dtype=np.int64)
        starts = list(range(0, len(ids), size_workload))
        ends = list(range(size_workload, len(ids), size_workload)) + [len(ids)]
        todo = [
            (s, e, ids[s:e])
            for s, e in zip(starts, ends)
            if not (dl_dir / f"{tissue}.{s:08d}-{e:08d}.h5ad").exists()
        ]
        if todo:
            remaining[tissue] = todo

    return remaining


# ---------------------------------------------------------------------------
# Per-tissue processing (streaming: flush complete chunks immediately)
# ---------------------------------------------------------------------------

def _process_tissue(
    output_dir: Path,
    version: str,
    tissue: str,
    chunks: List[Tuple[int, int, np.ndarray]],
) -> int:
    """Download all remaining chunks for one tissue using per-dataset queries.

    Memory-safe: chunks are written to disk (and evicted from RAM) as soon as
    all their cells have been received, so peak memory is bounded by the
    largest single dataset batch rather than the entire tissue.
    """
    import anndata as ad
    import scipy.sparse as sp

    dl_dir = output_dir / "download_dir"

    # Build lookup: soma_joinid → chunk key, and track which IDs each chunk needs
    soma_to_chunk: Dict[int, Tuple[int, int]] = {}
    chunk_needed: Dict[Tuple[int, int], Set[int]] = {}
    for s, e, obs_ids in chunks:
        key = (s, e)
        ids_set = set(obs_ids.tolist())
        chunk_needed[key] = ids_set
        for sid in ids_set:
            soma_to_chunk[int(sid)] = key

    # --- Step 1: metadata query (obs only, no expression matrix) ---
    all_needed_ids = np.concatenate([obs_ids for _, _, obs_ids in chunks])
    try:
        mapping_df = _get_obs_metadata(version, tissue, all_needed_ids)
    except Exception as exc:
        logger.error(f"[{tissue}] Metadata query failed: {exc}")
        return 0

    soma_to_dataset: Dict[int, str] = dict(
        zip(mapping_df["soma_joinid"].tolist(), mapping_df["dataset_id"].tolist())
    )

    # --- Step 2: load var target genes ---
    var_file = output_dir / "metadata_preparation_dir" / f"{tissue}.var.tsv"
    var_df = pd.read_csv(var_file, sep="\t", index_col=0)
    target_gene_ids: List[int] = var_df["soma_joinid"].tolist()

    # --- Step 3: group obs_ids by dataset_id ---
    dataset_to_obs: Dict[str, List[int]] = {}
    for sid, did in soma_to_dataset.items():
        dataset_to_obs.setdefault(did, []).append(int(sid))

    logger.info(
        f"[{tissue}] {len(chunks)} chunks via {len(dataset_to_obs)} datasets "
        f"(≤{_MAX_CELLS_PER_BATCH} cells/query)"
    )

    # chunk_buffer: {(start, end): {soma_joinid: (obs_dict, X_row, var_df)}}
    # Entries are deleted immediately after each chunk is written to disk.
    chunk_buffer: Dict[Tuple[int, int], Dict[int, Any]] = {
        (s, e): {} for s, e, _ in chunks
    }
    var_ref = None  # cached var DataFrame (same shape for all datasets)
    written = 0

    def _flush_complete_chunks() -> None:
        """Write any chunk whose buffer is now full, then evict it from RAM."""
        nonlocal written, var_ref
        for s, e, obs_ids in chunks:
            key = (s, e)
            buf = chunk_buffer.get(key)
            if buf is None:
                continue  # already written and evicted
            if len(buf) < len(obs_ids):
                continue  # still waiting for more cells

            ordered_ids = [int(x) for x in obs_ids]
            obs_rows = [buf[sid][0] for sid in ordered_ids]
            x_rows = [buf[sid][1] for sid in ordered_ids]

            obs_out = pd.DataFrame(obs_rows)
            obs_out.index = [str(i) for i in range(len(obs_out))]

            if sp.issparse(x_rows[0]):
                X = sp.vstack(x_rows)
            else:
                X = np.vstack(x_rows)

            if "raw_sum" not in obs_out.columns:
                obs_out["raw_sum"] = np.asarray(X.sum(axis=1)).flatten()

            vout = (var_ref if var_ref is not None else buf[ordered_ids[0]][2]).copy()
            vout.index = [str(i) for i in range(len(vout))]

            out_path = dl_dir / f"{tissue}.{s:08d}-{e:08d}.h5ad"
            ad.AnnData(X=X, obs=obs_out, var=vout).write_h5ad(out_path, compression="gzip")
            logger.info(f"[{tissue}] Written {out_path.name} ({len(obs_ids)} cells)")
            written += 1
            del chunk_buffer[key]  # evict immediately to free RAM

    # --- Step 4: per-dataset queries with sub-batching and streaming flush ---
    for dataset_id, ds_obs_ids in dataset_to_obs.items():
        for batch_start in range(0, len(ds_obs_ids), _MAX_CELLS_PER_BATCH):
            batch_ids = ds_obs_ids[batch_start: batch_start + _MAX_CELLS_PER_BATCH]
            adata = None
            for attempt in range(1, _MAX_RETRY + 1):
                try:
                    adata = _get_anndata_for_dataset(
                        version, dataset_id, batch_ids, target_gene_ids
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        f"[{tissue}/{dataset_id[:8]}] batch@{batch_start} "
                        f"attempt {attempt}/{_MAX_RETRY}: {exc}"
                    )
                    if attempt == _MAX_RETRY:
                        logger.error(
                            f"[{tissue}/{dataset_id[:8]}] all retries exhausted, "
                            f"skipping {len(batch_ids)} cells"
                        )
                    else:
                        time.sleep(10)

            if adata is None:
                continue

            logger.info(
                f"[{tissue}/{dataset_id[:8]}] received {adata.n_obs} cells, "
                f"{adata.n_vars} genes"
            )

            if var_ref is None:
                var_ref = adata.var.copy()

            obs_soma_ids = adata.obs["soma_joinid"].tolist()
            for i, sid in enumerate(obs_soma_ids):
                sid = int(sid)
                chunk_key = soma_to_chunk.get(sid)
                if chunk_key is None or chunk_key not in chunk_buffer:
                    continue  # chunk already written
                chunk_buffer[chunk_key][sid] = (
                    adata.obs.iloc[i].to_dict(),
                    adata.X[i, :],
                    var_ref,
                )
            del adata

            # Flush any newly completed chunks before loading the next batch
            _flush_complete_chunks()

    # Final flush pass (catches any chunk that became complete on the last batch)
    _flush_complete_chunks()

    # Warn about any chunks still incomplete (cells missing from census)
    for (s, e), buf in chunk_buffer.items():
        needed = chunk_needed[(s, e)]
        logger.warning(
            f"[{tissue}] chunk {s:08d}-{e:08d} incomplete: "
            f"{len(buf)}/{len(needed)} cells — skipping"
        )

    return written


# ---------------------------------------------------------------------------
# Public API — same signature as download.download()
# ---------------------------------------------------------------------------

def download(
    output_dir: str,
    version: str,
    num_worker: int,
    size_workload: int,
) -> None:
    """Drop-in replacement for download.download() using per-dataset queries.

    Processes tissues sequentially; within each tissue, dataset queries run
    sequentially to avoid SOMA rate-limiting.  num_worker is accepted for
    interface compatibility but currently unused at the tissue level.
    """
    output_path = Path(output_dir)
    output_path.joinpath("download_dir").mkdir(exist_ok=True, parents=True)

    remaining = _find_remaining_chunks(output_path, size_workload)
    if not remaining:
        logger.info("All chunks already downloaded.")
        return

    total_chunks = sum(len(v) for v in remaining.values())
    logger.info(
        f"download_by_dataset: {len(remaining)} tissues, "
        f"{total_chunks} remaining chunks"
    )
    for tissue, n in sorted(
        ((t, len(c)) for t, c in remaining.items()), key=lambda x: -x[1]
    ):
        logger.info(f"  {tissue}: {n} chunks remaining")

    total_written = 0
    for tissue, chunks in sorted(
        remaining.items(), key=lambda x: -len(x[1])
    ):
        logger.info(f"=== Processing tissue: {tissue} ({len(chunks)} chunks) ===")
        written = _process_tissue(output_path, version, tissue, chunks)
        total_written += written
        logger.info(f"[{tissue}] Done: {written}/{len(chunks)} chunks written")

    logger.info(f"download_by_dataset complete: {total_written}/{total_chunks} chunks written")
