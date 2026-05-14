"""CLI for the molecule_nat_lang evaluator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import MoleculeNatLangEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="molecule_nat_lang pair scoring")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("--modality", default="molecule_nat_lang")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--template", default="{caption}\n{smiles}")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Combined-length-stratified subsample (replaces df.head slicing).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Bootstrap resamples for perplexity 95%% CI (0 disables).",
    )
    parser.add_argument(
        "--predictions-preview-count",
        type=int,
        default=20,
        help="Per-evaluation best/worst-fit pairs shown in predictions.txt.",
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
    evaluator = MoleculeNatLangEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        pairs_path=Path(args.pairs_csv),
        config={
            "smiles_column": args.smiles_column,
            "caption_column": args.caption_column,
            "template": args.template,
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
