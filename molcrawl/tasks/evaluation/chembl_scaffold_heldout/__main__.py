"""CLI for ChEMBL scaffold held-out evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import ChEMBLScaffoldHeldoutEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChEMBL scaffold held-out evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("--modality", default="compounds")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heldout-csv", required=True)
    parser.add_argument("--train-csv", default=None, help="Required for encoder probe mode")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Length-stratified subsample (perplexity) or class-balanced subsample "
        "(probe). Replaces the legacy df.head() slicing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed for stratified subsample + bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Bootstrap resamples for perplexity / probe-metric 95%% CIs (0 disables).",
    )
    parser.add_argument(
        "--predictions-preview-count",
        type=int,
        default=30,
        help="Number of best- + worst-fit SMILES shown in predictions.txt.",
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
    evaluator = ChEMBLScaffoldHeldoutEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        heldout_path=Path(args.heldout_csv),
        config={
            "smiles_column": args.smiles_column,
            "label_column": args.label_column,
            "train_csv": args.train_csv,
            "max_examples": args.max_examples,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "predictions_preview_count": args.predictions_preview_count,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
