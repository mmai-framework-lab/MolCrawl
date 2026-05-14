"""CLI entry point for one TAPE sub-task."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .data_preparation import TASKS, get_spec
from .evaluator import TAPEEvaluator


def get_spec_names():
    return list(TASKS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TAPE protein encoder evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="esm2")
    parser.add_argument("--modality", default="protein_sequence")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", required=True, choices=get_spec_names())
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Task-aware stratified subsample size applied to every split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Bootstrap resamples for the active metric pack 95%% CIs (0 disables).",
    )
    parser.add_argument(
        "--predictions-preview-count",
        type=int,
        default=20,
        help="Per-evaluation rows shown in predictions.txt.",
    )
    parser.add_argument(
        "--contact-min-separation",
        type=int,
        default=24,
        help="Long-range threshold |i-j| for contact_prediction (default: 24).",
    )
    parser.add_argument(
        "--contact-pairs-per-protein",
        type=int,
        default=50,
        help="Positive (and equal-N negative) pairs sampled per training "
             "protein for contact_prediction logreg head (default: 50).",
    )
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
    evaluator = TAPEEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        task_dir=Path(args.task_dir),
        task_spec=get_spec(args.task),
        config={
            "max_examples": args.max_examples,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "predictions_preview_count": args.predictions_preview_count,
            "contact_min_separation": args.contact_min_separation,
            "contact_pairs_per_protein": args.contact_pairs_per_protein,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
