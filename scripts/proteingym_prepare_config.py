"""Register MolCrawl models in a ProteinGym config copy and in constants.json.

Issue #388, M2 / step 7. Writes ``<base>/configs/molcrawl.json`` -- a copy of the official
``config.json`` whose chosen ``model_list_*`` holds only the MolCrawl models we actually
run -- and registers each model's display name in the clone's ``constants.json``
(``clean_names`` / ``model_details`` / ``model_references``). References are left blank:
no citation is fabricated for an in-house model. A ``constants.json.orig`` backup is kept
and the added keys are recorded in ``<base>/configs/constants_molcrawl_additions.json`` so
the change to the vendored file is auditable.

    python scripts/proteingym_prepare_config.py \
        --base <.../eval/proteingym/official> \
        --category model_list_zero_shot_substitutions_DMS \
        --models MolCrawl_GPT2 MolCrawl_BERT \
        --key mutant

Re-running is idempotent: molcrawl.json is rewritten and constants entries are updated in
place (the backup is only created once, from the pristine clone).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ENTRY = {
    "input_score_name": "score",
    "directionality": 1,
    "model_type": "Protein language model",
}
DETAIL = ("MolCrawl protein-sequence language model (in-house). Zero-shot variant effect "
          "scored as the sequence log-likelihood difference mutant - reference; see the "
          "MolCrawl repository for training and scoring details.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Register MolCrawl models in a ProteinGym config")
    ap.add_argument("--base", required=True, help=".../eval/proteingym/official")
    ap.add_argument("--category", required=True,
                    help="config model_list key, e.g. model_list_zero_shot_substitutions_DMS")
    ap.add_argument("--models", nargs="+", required=True, help="registered model names")
    ap.add_argument("--key", default="mutant", help="merge key (mutant for subs, mutated_sequence for indels)")
    ap.add_argument("--replace", action="store_true",
                    help="set the category to ONLY these models (default: add to the official list)")
    args = ap.parse_args()

    base = Path(args.base)
    vendor = base / "vendor" / "ProteinGym"
    cfg_path = vendor / "config.json"
    const_path = vendor / "proteingym" / "constants.json"
    out_dir = base / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(cfg_path))
    if args.category not in cfg:
        raise SystemExit(f"category {args.category} not in config.json; keys: {list(cfg)}")
    entries = {m: {**ENTRY, "location": m, "key": args.key} for m in args.models}
    if args.replace:
        cfg[args.category] = entries
    else:
        cfg[args.category] = {**cfg[args.category], **entries}
    json.dump(cfg, open(out_dir / "molcrawl.json", "w"), indent=2)

    # constants.json: back up the pristine clone once, then register display names.
    backup = const_path.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy2(const_path, backup)
    const = json.load(open(const_path))
    additions = {"clean_names": {}, "model_details": {}, "model_references": {}}
    for m in args.models:
        const.setdefault("clean_names", {})[m] = m.replace("_", " ")
        const.setdefault("model_details", {})[m] = DETAIL
        const.setdefault("model_references", {})[m] = ""  # no fabricated citation
        additions["clean_names"][m] = m.replace("_", " ")
        additions["model_details"][m] = DETAIL
        additions["model_references"][m] = ""
    json.dump(const, open(const_path, "w"), indent=2)
    json.dump(additions, open(out_dir / "constants_molcrawl_additions.json", "w"), indent=2)

    print(f"molcrawl.json: {args.category} -> {list(entries)} (key={args.key}, "
          f"{'replace' if args.replace else 'add'})")
    print(f"constants.json: registered {args.models} (references blank); backup at {backup.name}")
    print(f"wrote {out_dir / 'molcrawl.json'}")


if __name__ == "__main__":
    main()
