"""CLI for ChEBI-20 bidirectional evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from molcrawl.tasks.evaluation._base import ModelHandle

from .evaluator import ChEBI20Evaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChEBI-20 bidirectional generation evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--arch", default="gpt2")
    parser.add_argument("--modality", default="molecule_nat_lang")
    parser.add_argument("--size", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direction", default="both", choices=("both", "mol_to_cap", "cap_to_mol"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    evaluator = ChEBI20Evaluator(
        handle=handle,
        output_dir=Path(args.output_dir),
        dataset_dir=Path(args.dataset_dir),
        config={
            "direction": args.direction,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "max_examples": args.max_examples,
        },
    )
    result = evaluator.run()
    print(f"metrics: {result.metrics}")
    print(f"report: {result.report_paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
