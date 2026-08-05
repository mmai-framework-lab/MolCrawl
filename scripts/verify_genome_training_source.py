"""Pre-launch check that a genome subset run reads the v2 (contig-split) data.

Two generations of this dataset exist, and older exports predating the
contig-unit split and the human chr21/22/X/Y hold-out may still be reachable
on a given deployment. Both generations use the same directory layout and the
same 21 subset names, so aiming ``LEARNING_SOURCE_DIR`` at an older export does
not fail — it silently trains on the wrong data.

Row counts do not separate the generations: the train split differs by well
under a percent, and valid/test are identical in size. The reliable
discriminator is the schema. Older exports carry only ``input_ids``, while v2
adds ``accession`` and ``contig_id`` because it needs them for the
contig-disjoint split.

Note that checking for the held-out human chromosomes does NOT work as a
*detector*: an older export has no ``contig_id`` column at all, so those
accessions can never appear and the check passes vacuously. It is kept below as
gate ③ for what it does verify — the contents of a dataset already confirmed to
be v2 by gate ①.

Gates:
  ① schema: ``accession`` and ``contig_id`` present in every split  (=> v2)
  ② splits: train/valid/test all present
  ③ chr exclusion: no NC_000021/22/23(X)/24(Y) window in any split

Run (``--source-dir`` defaults to ``$LEARNING_SOURCE_DIR/genome_sequence``, so
by default this validates the exact source the run is about to use):
  PYTHONPATH=<repo_root> python scripts/verify_genome_training_source.py \
      --subset mammal_centered --model gpt2
  # or check every subset/model the run will touch:
  PYTHONPATH=<repo_root> python scripts/verify_genome_training_source.py --all
"""

import argparse
import os
import sys

import pyarrow.compute as pc
from datasets import load_from_disk

# v2 markers. Pre-v2 exports carry only `input_ids`.
V2_COLUMNS = ("accession", "contig_id")
REQUIRED_SPLITS = ("train", "valid", "test")

# Human GRCh38 RefSeq accessions held out from pretraining. These prefixes are
# globally unique to human, so a prefix match cannot collide with other species.
EXCLUDED_CONTIG_PREFIXES = {
    "chr21": "NC_000021",
    "chr22": "NC_000022",
    "chrX": "NC_000023",
    "chrY": "NC_000024",
}

SUBSETS = (
    ["mammal_centered"]
    + [f"eukaryote_matched_random_seed{i}" for i in range(1, 11)]
    + [f"global_random_seed{i}" for i in range(1, 11)]
)


def _distinct_contigs(split):
    """Unique contig_id values, read chunk-wise so we never materialise a column."""
    seen = set()
    for chunk in split.data.column("contig_id").chunks:
        seen.update(pc.unique(chunk).to_pylist())
    return seen


def check(source_dir, subset, model, deep=True):
    """Return a list of failure strings; empty means the dataset passed."""
    path = f"{source_dir}/{subset}/training_ready_hf_dataset_{model}"
    failures = []
    try:
        dsd = load_from_disk(path)
    except Exception as exc:  # noqa: BLE001  (report, don't crash the sweep)
        return [f"{path}: load_from_disk failed: {exc}"]

    missing_splits = [s for s in REQUIRED_SPLITS if s not in dsd]
    if missing_splits:
        failures.append(f"{path}: missing split(s) {missing_splits}")

    for name in REQUIRED_SPLITS:
        if name not in dsd:
            continue
        columns = dsd[name].column_names
        absent = [c for c in V2_COLUMNS if c not in columns]
        if absent:
            # Gate ①. This is the old export; gate ③ below is meaningless
            # without contig_id, so skip it for this split.
            failures.append(
                f"{path}[{name}]: missing {absent} -> reading a pre-v2 export, "
                f"not v2 (columns: {columns})"
            )
            continue
        if not deep:
            continue
        contigs = _distinct_contigs(dsd[name])
        for label, prefix in sorted(EXCLUDED_CONTIG_PREFIXES.items()):
            hits = sorted(c for c in contigs if c.startswith(prefix))
            if hits:
                failures.append(
                    f"{path}[{name}]: {label} ({prefix}) present: {hits[:5]}"
                )
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=os.environ.get("LEARNING_SOURCE_DIR", "") + "/genome_sequence",
        help="genome_sequence dir; defaults to $LEARNING_SOURCE_DIR/genome_sequence",
    )
    parser.add_argument("--subset", help="single subset name (default: --all)")
    parser.add_argument("--model", choices=("gpt2", "bert"), help="single model")
    parser.add_argument("--all", action="store_true", help="check all 21 x 2")
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="run gates ① and ② only; skip the chr scan (fast)",
    )
    args = parser.parse_args()

    if not args.source_dir.strip("/"):
        parser.error("set LEARNING_SOURCE_DIR or pass --source-dir")

    if args.all or not args.subset:
        targets = [(s, m) for s in SUBSETS for m in ("gpt2", "bert")]
    else:
        models = [args.model] if args.model else ["gpt2", "bert"]
        targets = [(args.subset, m) for m in models]

    print(f"source: {args.source_dir}")
    all_failures = []
    for subset, model in targets:
        failures = check(args.source_dir, subset, model, deep=not args.schema_only)
        status = "OK" if not failures else "FAIL"
        print(f"  [{status}] {subset}/{model}")
        for line in failures:
            print(f"        {line}")
        all_failures.extend(failures)

    if all_failures:
        print(f"\n{len(all_failures)} failure(s). Do not launch.", file=sys.stderr)
        return 1
    print(f"\nall {len(targets)} dataset(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
