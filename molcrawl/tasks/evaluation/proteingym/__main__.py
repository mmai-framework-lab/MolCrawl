"""CLI entry point for the ProteinGym evaluator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import ProteinGymEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ProteinGym mutation effect evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("--modality", default="protein_sequence")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proteingym-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of variants scored from the CSV. "
            "Omit to evaluate on the full assay."
        ),
    )
    parser.add_argument(
        "--no-stratify-bin",
        dest="stratify_bin",
        action="store_false",
        help=(
            "Disable per-DMS-bin stratified sampling. When DMS_bin_score "
            "is present, stratified sampling is the default and keeps both "
            "functional / non-functional classes represented under small n."
        ),
    )
    parser.set_defaults(stratify_bin=True)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Bootstrap resamples for the 95 %% CI (0 disables).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="[deprecated] Legacy cap; re-interpreted as --n-examples.",
    )
    parser.add_argument(
        "--predictions-preview-count",
        type=int,
        default=20,
        help=(
            "Number of variants rendered in the predictions.txt narrative. "
            "Set 0 to skip the preview; predictions.jsonl is always produced."
        ),
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
    evaluator = ProteinGymEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        proteingym_path=Path(args.proteingym_data),
        config={
            "context_length": args.context_length,
            "n_examples": args.n_examples,
            "stratify_bin": args.stratify_bin,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "max_examples": args.max_examples,
            "predictions_preview_count": args.predictions_preview_count,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
