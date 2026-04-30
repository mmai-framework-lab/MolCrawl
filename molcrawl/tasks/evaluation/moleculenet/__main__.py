"""CLI entry point for one MoleculeNet sub-task."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .data_preparation import get_task
from .evaluator import MoleculeNetEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoleculeNet evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="chemberta2")
    parser.add_argument("--modality", default="compounds")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subtask", required=True, help="e.g. bbbp, tox21, esol")
    parser.add_argument("--task-dir", required=True, help="Directory with raw.csv + manifest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="scaffold", choices=("scaffold", "random"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    handle = ModelHandle(
        arch=args.arch,
        modality=args.modality,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        size=args.size,
        extras={"device": args.device},
    )
    evaluator = MoleculeNetEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        task_dir=Path(args.task_dir),
        task_spec=get_task(args.subtask),
        config={
            "split": args.split,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "max_examples": args.max_examples,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
