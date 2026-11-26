#!/usr/bin/env python3
"""Collect the eight ChEBI-20 retrieval runs into the order's attachment tables.

The 2026-11-26 order, items 5 and 7. Per run directory (written by
scripts/molnl_chebi20_retrieval.py) this reads ranks.tsv, finetune_curve.tsv and
run_manifest.json, and writes

  rt1-molnl-chebi20-ranks-<stamp>.tsv            every test description's rank, 8 runs
  rt1-molnl-chebi20-finetune-curves-<stamp>.tsv  every step's training loss, and the
                                                 valid loss at every evaluated step
  rt1-molnl-chebi20-metrics-<stamp>.tsv          Hits@1, Hits@10, MRR, Mean Rank per run,
                                                 next to chance and Text2Mol's numbers
  rt1-molnl-chebi20-overlap-<stamp>.tsv          the test pairs item 6 found in the corpus

The four metrics are computed here again from the combined ranks file, not copied from
the runs' metrics.json, and the two are compared: the ranks file must be enough to give
them, which is what item 7 asks of it.

    python scripts/molnl_chebi20_report.py --arms <arms tsv> --runs <dir> \
        --overlap-items <tsv> --out-dir <dir> --stamp YYYYMMDD
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone

BASELINES = [
    # name, hits@1, hits@10, mrr, mean_rank, source
    ("chance (1 of 33,010)", 1 / 33010, 10 / 33010, None, (33010 + 1) / 2,
     "Hits@1 = 1/33,010; Hits@10 = 10/33,010; mean rank (N+1)/2; MRR = H(33,010)/33,010"),
    ("Text2Mol single model (paper)", None, None, 0.372, None, "order item 5"),
    ("Text2Mol ensemble (paper)", 0.344, 0.811, 0.499, 20.21, "order item 5"),
]


def read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader((line for line in fh if not line.startswith("#")),
                                   delimiter="\t"))


def harmonic(n):
    return sum(1.0 / k for k in range(1, n + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--runs", required=True, help="the directory holding one dir per arm")
    ap.add_argument("--overlap-items", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stamp", required=True)
    a = ap.parse_args()

    stamp_time = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    arms = read_tsv(a.arms)
    out = lambda name: os.path.join(a.out_dir, f"rt1-molnl-chebi20-{name}-{a.stamp}.tsv")  # noqa: E731

    rank_rows, curve_rows, metric_rows = [], [], []
    for arm in arms:
        d = os.path.join(a.runs, arm["arm"])
        man = json.load(open(os.path.join(d, "run_manifest.json")))
        if man.get("max_steps_override") is not None:
            raise SystemExit(f"{d}: a smoke test (max_steps {man['max_steps_override']})")
        if man["steps_run"] != man["total_steps"]:
            raise SystemExit(f"{d}: ran {man['steps_run']} of {man['total_steps']} steps")
        ranks = read_tsv(os.path.join(d, "ranks.tsv"))
        rank_rows += ranks
        curve_rows += read_tsv(os.path.join(d, "finetune_curve.tsv"))
        r = [int(x["rank"]) for x in ranks]
        n = len(r)
        m = {"hits@1": sum(x <= 1 for x in r) / n, "hits@10": sum(x <= 10 for x in r) / n,
             "mrr": sum(1 / x for x in r) / n, "mean_rank": sum(r) / n}
        for k, v in m.items():
            if abs(v - man["test"][k]) > 1e-9:
                raise SystemExit(f"{d}: {k} from ranks.tsv {v} != run's {man['test'][k]}")
        metric_rows.append({
            "row": arm["arm"], "arch": arm["arch"], "size": arm["size"],
            "hits@1": f"{m['hits@1']:.4f}", "hits@10": f"{m['hits@10']:.4f}",
            "mrr": f"{m['mrr']:.4f}", "mean_rank": f"{m['mean_rank']:.2f}",
            "n_queries": n, "n_candidates": ranks[0]["n_candidates"],
            "queries_with_ties": sum(int(x["ties"]) > 0 for x in ranks),
            "selected_step": man["selected_step"], "total_steps": man["total_steps"],
            "selected_valid_loss": f"{man['selected_valid_loss']:.6f}",
            "source": f"{arm['arm']}/ranks.tsv (job {man['job_id']}, commit {man['git']['commit']})",
        })
    chance_mrr = harmonic(33010) / 33010
    for name, h1, h10, mrr, mr, src in BASELINES:
        if name.startswith("chance"):
            mrr = chance_mrr
        metric_rows.append({
            "row": name, "arch": "", "size": "",
            "hits@1": f"{h1:.6f}" if h1 is not None else "",
            "hits@10": f"{h10:.6f}" if h10 is not None else "",
            "mrr": f"{mrr:.4f}" if mrr is not None and mrr > 0.01 else (f"{mrr:.6f}" if mrr else ""),
            "mean_rank": f"{mr:.2f}" if mr is not None else "",
            "n_queries": "", "n_candidates": "33010", "queries_with_ties": "",
            "selected_step": "", "total_steps": "", "selected_valid_loss": "", "source": src,
        })

    def write(path, rows, header_lines):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("#! report-tsv 1\n")
            for h in header_lines:
                fh.write(f"# {h}\n")
            fh.write(f"# generated: {stamp_time} by scripts/molnl_chebi20_report.py\n")
            cols = list(rows[0])
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    write(out("ranks"), rank_rows, [
        "model: the eight runs of rt1-molnl-chebi20-arms (column arm)",
        "source: each run's ranks.tsv; one row per ChEBI-20 test description and run",
        "note: description_id is the ChEBI-20 CID of the description; the correct molecule is the same CID. "
        "rank = 1 + the number of other candidates with equal or higher cosine similarity (ties count against the model); "
        "ties = how many other candidates scored exactly the same",
        "note: Hits@k = share of rows with rank <= k; MRR = mean of 1/rank; Mean Rank = mean of rank, per arm",
    ])
    write(out("finetune-curves"), curve_rows, [
        "model: the eight runs of rt1-molnl-chebi20-arms (column arm)",
        "source: each run's finetune_curve.tsv",
        "note: train_loss at every step (symmetric InfoNCE on the global batch of 256); eval_loss on ChEBI-20 valid "
        "(3,301 pairs, file order, blocks of 256) at step 0 and every 50 steps, blank where not evaluated; "
        "eval_mrr_valid3301 = MRR of each valid description against the 3,301 valid molecules",
    ])
    write(out("metrics"), metric_rows, [
        "model: the eight runs plus chance and Text2Mol's published numbers",
        "source: Hits@1, Hits@10, MRR and Mean Rank recomputed from rt1-molnl-chebi20-ranks; Text2Mol rows from the order",
    ])
    items = read_tsv(a.overlap_items)
    hit = [r for r in items if r["class"] != "neither"]
    write(out("overlap"), hit, [
        "model: none -- ChEBI-20 test pairs against the molnl pretraining corpus (train split of the packed build)",
        "source: scripts/molnl_chebi20_overlap.py",
        f"note: {len(hit)} of {len(items)} test pairs; class is pair / description_and_molecule_apart / "
        "description_only / molecule_only. Matching rules are in the script's docstring and the order's answer §6",
    ])
    for r in metric_rows:
        print("\t".join(str(r[c]) for c in ("row", "hits@1", "hits@10", "mrr", "mean_rank")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
