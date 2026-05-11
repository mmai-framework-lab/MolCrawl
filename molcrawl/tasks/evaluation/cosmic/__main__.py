"""CLI for COSMIC evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import CosmicEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COSMIC pathogenicity evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("--modality", default="genome_sequence")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cosmic-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="FATHMM_PREDICTION")
    parser.add_argument("--context-length", type=int, default=512)
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
    evaluator = CosmicEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        cosmic_path=Path(args.cosmic_data),
        config={
            "label_column": args.label_column,
            "context_length": args.context_length,
            "max_examples": args.max_examples,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
