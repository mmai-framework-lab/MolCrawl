"""CLI entry point for the ClinVar evaluator.

Invoke as::

    python -m molcrawl.tasks.evaluation.clinvar \\
        --model-path path/to/ckpt.pt \\
        --tokenizer-path path/to/tokenizer.model \\
        --clinvar-data path/to/clinvar.csv \\
        --output-dir experiment_data/eval/clinvar_smoke
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import ClinVarEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ClinVar pathogenicity evaluation")
    parser.add_argument("--model-path", required=True, help="Trained checkpoint path")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="SentencePiece tokenizer path (required for GPT-2 / BERT genome models)",
    )
    parser.add_argument("--clinvar-data", required=True, help="ClinVar table (CSV/TSV/JSON)")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for metrics.json and REPORT.md",
    )
    parser.add_argument("--arch", default="gpt2", help="Model architecture (default: gpt2)")
    parser.add_argument(
        "--modality",
        default="genome_sequence",
        help="Foundation model modality (default: genome_sequence)",
    )
    parser.add_argument("--size", default=None, help="Optional size tag (small/medium/large/xl)")
    parser.add_argument("--device", default="cuda", help="Device string (default: cuda)")
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Maximum context length fed to the adapter",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Cap the number of variants evaluated (useful for smoke runs)",
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

    evaluator = ClinVarEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        clinvar_path=args.clinvar_data,
        config={
            "context_length": args.context_length,
            "max_examples": args.max_examples,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
