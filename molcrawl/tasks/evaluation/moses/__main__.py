"""CLI entry point for MOSES evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import MOSESEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MOSES generation-quality evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("--modality", default="compounds")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--reference-limit", type=int, default=None)
    parser.add_argument("--disable-extended", action="store_true")
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
    evaluator = MOSESEvaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        reference_dir=Path(args.reference_dir),
        config={
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "reference_limit": args.reference_limit,
            "enable_extended_metrics": not args.disable_extended,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
