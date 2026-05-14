"""CLI for gnomAD allele-frequency correlation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import GnomadAFEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gnomAD allele frequency correlation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("--modality", default="genome_sequence")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gnomad-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--n-per-bin",
        type=int,
        default=None,
        help=(
            "AF-log-bin stratified sample size per bin. Omit to evaluate on "
            "the full dataset."
        ),
    )
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
        help=(
            "[deprecated] Legacy total-row cap; re-interpreted as "
            "n_per_bin = max_examples // 6."
        ),
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
    evaluator = GnomadAFEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        gnomad_path=Path(args.gnomad_data),
        config={
            "context_length": args.context_length,
            "n_per_bin": args.n_per_bin,
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
