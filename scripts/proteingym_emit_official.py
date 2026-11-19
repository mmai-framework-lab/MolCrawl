"""Emit ProteinGym per-assay prediction CSVs in the official input format.

Issue #388, M2. The MolCrawl sweep writes one ``predictions.jsonl`` per assay under
``<eval>/<slug>/proteingym_<DMS_id>_sweep/`` with, per variant, the fields ``mutant``,
``mutated_sequence`` and ``score`` (= mut_ll - ref_ll). The official ``merge.py`` expects,
per registered model, one ``<DMS_id>.csv`` under ``<scores>/<location>/`` carrying the
merge key (``mutant`` for substitutions, ``mutated_sequence`` for indels) and the score
column named by the model's ``input_score_name`` (we use ``score``).

This converts one slug's predictions into that layout under a chosen model name, so a model
can be scored under any registered name without baking a size choice into the code.

    python scripts/proteingym_emit_official.py \
        --slug protein_sequence-bert-small-s42 \
        --model-name MolCrawl_BERT \
        --mutation-type substitutions \
        --out <scores>/zero_shot/substitutions

Default eval root: <repo>/learning_source_20260730_protein_uniref50/experiment_data/eval
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The scored outputs live in the main checkout, not a worktree, so prefer an explicit
# LEARNING_SOURCE_DIR (as the workflows set) and fall back to the in-repo path.
_LSD = os.environ.get("LEARNING_SOURCE_DIR")
DEFAULT_EVAL = (Path(_LSD) / "experiment_data/eval" if _LSD
                else REPO / "learning_source_20260730_protein_uniref50/experiment_data/eval")


def emit(slug: str, model_name: str, mutation_type: str, eval_root: Path, out: Path) -> dict:
    dest = out / model_name
    dest.mkdir(parents=True, exist_ok=True)
    key = "mutant" if mutation_type == "substitutions" else "mutated_sequence"
    sweeps = sorted(glob.glob(str(eval_root / slug / "proteingym_*_sweep")))
    written, skipped, n_variants = 0, [], 0
    for sw in sweeps:
        dms_id = os.path.basename(sw)[len("proteingym_"):-len("_sweep")]
        jf = Path(sw) / "predictions.jsonl"
        if not jf.exists():
            skipped.append(f"{dms_id}: no predictions.jsonl")
            continue
        rows = []
        with open(jf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if key not in r or "score" not in r:
                    skipped.append(f"{dms_id}: missing {key}/score")
                    rows = []
                    break
                rows.append((r.get("mutant", ""), r["mutated_sequence"], r["score"]))
        if not rows:
            continue
        with open(dest / f"{dms_id}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["mutant", "mutated_sequence", "score"])
            w.writerows(rows)
        written += 1
        n_variants += len(rows)
    return {"model": model_name, "slug": slug, "assays_written": written,
            "variants": n_variants, "skipped": skipped, "dest": str(dest), "merge_key": key}


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit official-format ProteinGym prediction CSVs")
    ap.add_argument("--slug", required=True, help="MolCrawl eval slug, e.g. protein_sequence-bert-small-s42")
    ap.add_argument("--model-name", required=True, help="registered model name, e.g. MolCrawl_BERT")
    ap.add_argument("--mutation-type", choices=["substitutions", "indels"], default="substitutions")
    ap.add_argument("--eval-root", default=str(DEFAULT_EVAL))
    ap.add_argument("--out", required=True, help="scores dir; files land under <out>/<model-name>/")
    args = ap.parse_args()
    res = emit(args.slug, args.model_name, args.mutation_type,
               Path(args.eval_root), Path(args.out))
    print(f"{res['model']} <- {res['slug']}: {res['assays_written']} assays, "
          f"{res['variants']} variants, key={res['merge_key']} -> {res['dest']}")
    for s in res["skipped"][:20]:
        print(f"  skip {s}")
    if res["assays_written"] == 0:
        raise SystemExit("no assays emitted")


if __name__ == "__main__":
    main()
