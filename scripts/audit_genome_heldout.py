"""Check the held-out human chromosomes never reached the training data.

The downstream ClinVar evaluation scores variants on chr21/22/X/Y, so a model
that saw those contigs during pretraining is being asked to predict variants in
sequence it already read. That is a separate failure from anything about the
pretraining comparison, and it cannot be fixed after the fact -- only the
evaluation set can move.

The hold-out is defined by RefSeq accession prefix (NC_000021 .. NC_000024),
which names human chromosomes and nothing else. Other species are not excluded
and are not meant to be; ClinVar is human. This reports which species carry
those accessions at all, so "no hits" is distinguishable from "the human genome
is not in this subset".

Reports per subset and split: the accessions present, whether any held-out
contig appears, and the row count involved.
"""

import argparse
import glob
import json
import os
from collections import Counter

from molcrawl.data.genome_sequence.dataset.refseq.chr22_holdout import (
    HELDOUT_CONTIG_PREFIXES,
)


def audit_split(ds):
    """Held-out contigs present, and the rows they account for."""
    contigs = Counter(zip(ds["accession"], ds["contig_id"]))
    hits = {}
    for name, prefix in sorted(HELDOUT_CONTIG_PREFIXES.items()):
        matched = {k: n for k, n in contigs.items() if str(k[1]).startswith(prefix)}
        if matched:
            hits[name] = matched
    return contigs, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="genome_sequence dir")
    ap.add_argument("--models", default="gpt2,bert")
    ap.add_argument("--subsets", help="comma separated; default all")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    from datasets import load_from_disk

    models = args.models.split(",")
    subsets = (args.subsets.split(",") if args.subsets
               else sorted(os.path.basename(d.rstrip("/"))
                           for d in glob.glob(f"{args.root}/*/")
                           if glob.glob(f"{d}/training_ready_hf_dataset_*")))

    report, dirty = [], 0
    print(f"  {'subset':<34}{'model':>6}{'split':>7}{'rows':>14}{'accessions':>12}{'held-out':>10}")
    for s in subsets:
        for m in models:
            d = f"{args.root}/{s}/training_ready_hf_dataset_{m}"
            if not os.path.isdir(d):
                continue
            ds = load_from_disk(d)
            for split in ds:
                contigs, hits = audit_split(ds[split])
                accs = {k[0] for k in contigs}
                bad = sum(sum(v.values()) for v in hits.values())
                dirty += bool(hits)
                report.append({"subset": s, "model": m, "split": split,
                               "rows": len(ds[split]), "accessions": len(accs),
                               "heldout_rows": bad,
                               "heldout": {k: sorted(f"{a}/{c}" for a, c in v)
                                           for k, v in hits.items()}})
                print(f"  {s:<34}{m:>6}{split:>7}{len(ds[split]):>14,}{len(accs):>12}"
                      f"{('CLEAN' if not hits else f'{bad:,} rows'):>10}", flush=True)

    print(f"\n  splits with held-out contigs: {dirty} / {len(report)}")
    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=2)
        print(f"  wrote {args.json_out}")
    raise SystemExit(1 if dirty else 0)


if __name__ == "__main__":
    main()
