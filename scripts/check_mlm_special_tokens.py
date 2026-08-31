"""Report the MLM special-token ids each modality's BERT config actually resolves.

Two failures this answers, both raised in the 2026-08-28 verdict:

- ``mask_token_id`` is read off the tokenizer object with ``getattr(..., None)``
  (models/bert/main.py), so a tokenizer that does not expose it yields None and the
  run continues on the blended ``eval_loss``. Whether that can happen is a property
  of each modality's tokenizer, not of the training code.
- ``sep_token_id`` took the same shape twice already (PR #118, PR #120), where a
  mismatch silently disabled document masking.

Loading the config module is what the trainer itself does, so the ids reported here
are the ids a run would get. Nothing is trained; this is a read-only check.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path

# One representative BERT config per modality. Each builds its tokenizer a different
# way (a repo Tokenizer class, AutoTokenizer over a local dir, a factory function),
# which is exactly why this has to execute them rather than pattern-match the source.
CONFIGS = {
    "compounds": "molcrawl/tasks/pretrain/configs/compounds/bert_small.py",
    "genome_sequence": "molcrawl/tasks/pretrain/configs/genome_sequence/bert_small.py",
    "protein_sequence": "molcrawl/tasks/pretrain/configs/protein_sequence/bert_small.py",
    "rna": "molcrawl/tasks/pretrain/configs/rna/bert_small.py",
    "molecule_nat_lang": "molcrawl/tasks/pretrain/configs/molecule_nat_lang/bert_small.py",
}


def _unwrap(tok):
    """Mirror main.py: fall back to the inner tokenizer when the outer lacks the id."""
    if tok is not None and not hasattr(tok, "mask_token_id"):
        return getattr(tok, "tokenizer", None)
    return tok


def inspect(path: Path) -> dict:
    ns = runpy.run_path(str(path))
    tok = ns.get("actual_tokenizer") or ns.get("tokenizer")
    resolved = _unwrap(tok)
    out = {
        "tokenizer_class": type(tok).__name__ if tok is not None else None,
        "resolved_class": type(resolved).__name__ if resolved is not None else None,
        # getattr with a None default is what main.py does; keep the same semantics
        # so a missing attribute shows up here exactly as it would in a run.
        "mask_token_id": getattr(resolved, "mask_token_id", None),
        "sep_token_id": getattr(resolved, "sep_token_id", None),
        "mask_token": getattr(resolved, "mask_token", None),
        "vocab_size": getattr(resolved, "vocab_size", None),
        "document_masking": ns.get("document_masking"),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", help="write the results here as JSON")
    args = ap.parse_args()

    results, failures = {}, 0
    for modality, rel in CONFIGS.items():
        path = Path(rel)
        if not path.exists():
            results[modality] = {"error": f"config not found: {rel}"}
            failures += 1
            print(f"{modality:20s} SKIP  {rel} does not exist")
            continue
        try:
            info = inspect(path)
        except Exception as exc:  # a config that cannot load is itself the finding
            results[modality] = {"error": f"{type(exc).__name__}: {exc}"}
            failures += 1
            print(f"{modality:20s} ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        results[modality] = info
        ok = "OK  " if info["mask_token_id"] is not None else "NONE"
        if info["mask_token_id"] is None:
            failures += 1
        print(
            f"{modality:20s} {ok} mask_token_id={info['mask_token_id']!r} "
            f"mask_token={info['mask_token']!r} sep_token_id={info['sep_token_id']!r} "
            f"vocab={info['vocab_size']!r} tokenizer={info['resolved_class']}"
        )

    print()
    usable = sum(1 for v in results.values() if v.get("mask_token_id") is not None)
    print(f"mask_token_id resolved for {usable} / {len(CONFIGS)} modalities")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
