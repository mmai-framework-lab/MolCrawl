"""What a packed corpus actually wrote between documents, and whether masking keys on it.

``document_masking`` confines attention to one document inside a packed block by
splitting on a separator id. Which id that is belongs to the packing, not to the
tokenizer, and the two have disagreed before: RNA separates cells with token 0
while its tokenizer resolves ``sep_token_id`` to 25428, an id the data never
contains. The collator found zero boundaries, passed every batch through
unchanged, and the run looked entirely normal.

models/bert/main.py now refuses to start when the id it resolved never occurs, so
the silent no-op is gone -- but the refusal happens after the job has queued, a
node has been allocated and the dataset has loaded. This answers the same
question first, and answers the follow-up the refusal can only ask: if the
resolved id is not there, what is?

Read-only. Nothing is trained and nothing is written.
"""

from __future__ import annotations

import argparse
import collections
import json
import runpy
import sys
from pathlib import Path

# main.py keeps its globals inside `if __name__ == "__main__"`, so importing it
# runs nothing and the scan helper can be shared rather than reimplemented.
from molcrawl.models.bert.main import _scan_boundary_id


def _unwrap(tok):
    """Mirror main.py: fall back to the inner tokenizer when the outer lacks the id."""
    if tok is not None and not hasattr(tok, "mask_token_id"):
        return getattr(tok, "tokenizer", None)
    return tok


def _load_split(dataset_dir: str, split: str):
    """The arrow split main.py's legacy loader would read for this directory."""
    from datasets import load_from_disk

    root = Path(dataset_dir)
    candidates = [root / f"{split}.arrow", root / split]
    for path in candidates:
        if path.exists():
            return load_from_disk(str(path)), str(path)
    # Standard DatasetDict layout
    dataset = load_from_disk(str(root))
    return dataset[split], f"{root}[{split}]"


def _id_histogram(dataset, n_rows: int, top: int = 12):
    """The most common ids in the sampled rows, to name the separator that is there."""
    counter = collections.Counter()
    rows = 0
    total = min(int(n_rows), len(dataset))
    for i in range(total):
        counter.update(int(t) for t in dataset[i]["input_ids"])
        rows += 1
    return rows, counter.most_common(top), sum(counter.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="BERT config the run would use")
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows", type=int, default=2000,
                        help="rows to sample; main.py's own check uses 2000")
    parser.add_argument("--json", help="write the result here as well as printing it")
    args = parser.parse_args(argv)

    ns = runpy.run_path(args.config)
    tokenizer = _unwrap(ns.get("actual_tokenizer") or ns.get("tokenizer"))
    dataset_dir = ns.get("dataset_dir")
    if not dataset_dir:
        print(f"{args.config} sets no dataset_dir", file=sys.stderr)
        return 2

    # Same precedence main.py uses: an explicit config value wins, tokenizer is fallback.
    configured = ns.get("boundary_token_id")
    if configured is not None:
        boundary_id, source = int(configured), "config.boundary_token_id"
    else:
        boundary_id, source = getattr(tokenizer, "sep_token_id", None), "tokenizer.sep_token_id"

    print(f"config          : {args.config}")
    print(f"document_masking: {bool(ns.get('document_masking', False))}")
    print(f"dataset_dir     : {dataset_dir}")
    print(f"boundary id     : {boundary_id} (from {source})")
    for name in ("mask_token_id", "cls_token_id", "pad_token_id", "sep_token_id"):
        print(f"  {name:<14}: {getattr(tokenizer, name, None)}")

    dataset, loaded_from = _load_split(str(dataset_dir), args.split)
    print(f"split           : {args.split} ({len(dataset):,} rows) from {loaded_from}")

    stats = (_scan_boundary_id(dataset, boundary_id, n_rows=args.rows)
             if boundary_id is not None
             else {"rows": 0, "positions": 0, "count": 0, "rate": 0.0, "per_block": 0.0})
    print(f"\nboundary scan   : id {boundary_id} occurs {stats['count']:,} times "
          f"in {stats['positions']:,} positions ({stats['rate']:.4%}), "
          f"{stats['per_block']:.2f} per block over {stats['rows']:,} rows")

    rows, common, positions = _id_histogram(dataset, args.rows)
    print(f"\nmost common ids over the same {rows:,} rows ({positions:,} positions):")
    for token_id, count in common:
        marker = "  <- resolved boundary" if token_id == boundary_id else ""
        piece = None
        if tokenizer is not None:
            try:
                piece = tokenizer.convert_ids_to_tokens([token_id])[0]
            except Exception:
                piece = None
        print(f"  {token_id:>8}  {count:>12,}  {count / positions:7.3%}  "
              f"{piece or ''}{marker}")

    verdict = "ok" if stats["count"] else "absent"
    if not stats["count"]:
        print(f"\nid {boundary_id} does not occur in {rows:,} rows. Document masking would "
              "be a no-op, and main.py refuses to start. Set boundary_token_id to "
              "whatever the packer wrote -- the histogram above is where to look.")
    else:
        print(f"\nDocument masking has something to key on: {stats['per_block']:.2f} "
              f"boundaries per 1024-token block.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "config": args.config,
            "dataset_dir": str(dataset_dir),
            "split": args.split,
            "boundary_id": boundary_id,
            "boundary_source": source,
            "verdict": verdict,
            "scan": stats,
            "most_common": [[int(i), int(c)] for i, c in common],
        }, indent=2), encoding="utf-8")

    return 0 if stats["count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
