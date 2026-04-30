"""CLI for foldability proxies."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import ProteinFoldabilityEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protein foldability proxies")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("--modality", default="protein_sequence")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=None)
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
    evaluator = ProteinFoldabilityEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        reference_fasta=Path(args.reference_fasta),
        config={
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
