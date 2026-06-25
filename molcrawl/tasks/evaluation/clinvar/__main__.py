"""CLI entry point for the ClinVar evaluator.

Invoke as::

    python -m molcrawl.tasks.evaluation.clinvar \\
        --model-path path/to/ckpt.pt \\
        --tokenizer-path path/to/tokenizer.model \\
        --clinvar-data path/to/clinvar.csv \\
        --output-dir experiment_data/eval/clinvar_smoke \\
        --n-per-class 1000
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
        "--n-per-class",
        type=int,
        default=None,
        help=(
            "Class-balanced sample size per class (pathogenic and benign). "
            "Omit to evaluate on the full dataset."
        ),
    )
    parser.add_argument(
        "--no-stratify-chrom",
        dest="stratify_chrom",
        action="store_false",
        help=(
            "Disable per-chromosome stratified sampling within each class. "
            "Stratification is on by default to neutralise the per-chromosome "
            "pathogenic-rate variance (chrY ≈ 86 %%, chrX ≈ 48 %%, overall 27 %%)."
        ),
    )
    parser.set_defaults(stratify_chrom=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "[deprecated] Legacy cap on evaluated variants. When set without "
            "--n-per-class it is re-interpreted as n_per_class=max_examples//2 "
            "so old smoke scripts still draw both classes."
        ),
    )
    parser.add_argument(
        "--predictions-preview-count",
        type=int,
        default=20,
        help=(
            "Number of variants rendered in the human-readable "
            "predictions.txt narrative (sampled across {label} × {correct} "
            "quadrants). Set 0 to skip the preview; predictions.jsonl is "
            "always produced."
        ),
    )
    parser.add_argument(
        "--score-window-half",
        type=int,
        default=None,
        help=(
            "When set, the PLL average is restricted to a window of "
            "±N tokens around the variant centre (the model still "
            "sees full context). Default = full-sequence average, "
            "matching the historical behaviour. Sensible value for "
            "the 128-nt window produced by download_clinvar_sequences "
            "is 32 (= 65-token window around the variant)."
        ),
    )
    parser.add_argument(
        "--flank",
        type=int,
        default=64,
        help=(
            "Position of the variant centre within each input token "
            "sequence (default 64, matching the upstream window "
            "extraction in download_clinvar_sequences). Only used when "
            "--score-window-half is set."
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

    evaluator = ClinVarEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        clinvar_path=args.clinvar_data,
        config={
            "context_length": args.context_length,
            "n_per_class": args.n_per_class,
            "stratify_chrom": args.stratify_chrom,
            "seed": args.seed,
            "max_examples": args.max_examples,
            "predictions_preview_count": args.predictions_preview_count,
            "score_window_half": args.score_window_half,
            "flank": args.flank,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
