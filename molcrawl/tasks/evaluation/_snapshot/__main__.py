"""CLI for building the cross-task evaluation snapshot.

Invoke as::

    python -m molcrawl.tasks.evaluation._snapshot \\
        --input-dir experiment_data/eval \\
        --output-dir docs/evaluation \\
        [--previous docs/evaluation/snapshot_20260421.json]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .aggregator import (
    build_snapshot,
    collect_results,
    load_snapshot,
    write_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the cross-task evaluation snapshot")
    parser.add_argument("--input-dir", required=True, help="Root of per-task metrics.json files")
    parser.add_argument("--output-dir", required=True, help="Where snapshot_<date>.{json,md} is written")
    parser.add_argument("--previous", default=None, help="Path to a previous snapshot_*.json for diffing")
    parser.add_argument("--name", default=None, help="Override the snapshot file basename")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    entries = collect_results(Path(args.input_dir))
    snapshot = build_snapshot(entries)
    previous = load_snapshot(Path(args.previous)) if args.previous else None
    paths = write_snapshot(
        snapshot,
        output_dir=Path(args.output_dir),
        previous_snapshot=previous,
        name=args.name,
    )
    print("snapshot written:")
    for key, value in paths.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
