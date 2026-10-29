"""ProteinGym per-assay and per-model tables for the 8 protein models.

protein-order-2026-10-23 §1 (per-assay, one row per assay x model, incomplete rows
included with status) and §2 (per-model summary). Built from files only (no GPU):
the per-assay metrics.json under each model's eval slug, the scored checkpoint's
pretrain loss (BERT eval_loss_mask from trainer_state; GPT-2 val_loss from logging csv),
and the ProteinGym reference (per-assay total mutant count + set metadata for §3).

Run:  python scripts/protein_proteingym_tables.py <out_dir>
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_PROD = REPO / "learning_source_protein_runs/protein_sequence"
RUNS_U50 = REPO / "learning_source_20260730_protein_uniref50/protein_sequence/runs"
EVAL = REPO / "learning_source_20260730_protein_uniref50/experiment_data/eval"
PGDIR = REPO / "learning_source_20260730_protein_uniref50/eval/proteingym/unpacked/DMS_ProteinGym_substitutions"
REF = REPO / "learning_source_20260625-newBERT/protein_sequence/proteingym/proteingym_v1.3/DMS_substitutions.csv"

# arch, size, lr, train seed, eval slug, pretrain-loss source dir, loss metric, kind
MODELS = [
    ("bert", "small", "3e-4", 42, "protein_sequence-bert-small-best", RUNS_PROD / "bert-output/protein_sequence-small-lr3e4", "eval_loss_mask", "bert"),
    ("bert", "medium", "1e-4", 42, "protein_sequence-bert-medium-best", RUNS_PROD / "bert-output/protein_sequence-medium-lr1e4", "eval_loss_mask", "bert"),
    ("bert", "large", "3.5e-5", 42, "protein_sequence-bert-large-best", RUNS_PROD / "bert-output/protein_sequence-large-lr3p5e5", "eval_loss_mask", "bert"),
    ("bert", "xl", "2.4e-5", 42, "protein_sequence-bert-xl-best", RUNS_PROD / "bert-output/protein_sequence-xl-lr2p4e5", "eval_loss_mask", "bert"),
    ("gpt2", "small", "1.2e-3", 1001, "protein_sequence-gpt2-small-best", RUNS_PROD / "gpt2-output/protein_sequence-small-lr1p2e3", "val_loss", "gpt2"),
    ("gpt2", "medium", "6e-4", 1001, "protein_sequence-gpt2-medium-best", RUNS_U50 / "retrain2-gpt2-medium-lr6e-4-s1001", "val_loss", "gpt2"),
    ("gpt2", "large", "6e-4", 1001, "protein_sequence-gpt2-large-best", RUNS_U50 / "retrain2-gpt2-large-lr6e-4-s1001", "val_loss", "gpt2"),
    ("gpt2", "xl", "6e-4", 1001, "protein_sequence-gpt2-xl-best", RUNS_U50 / "retrain2-gpt2-xl-lr6e-4-s1001", "val_loss", "gpt2"),
]

# protein-judgment-2026-10-29 §1/§3: BERT xl never learned in pretraining, so its
# downstream was cut off (option c). Its ProteinGym spearman is not a confirmed value
# and must not appear in the tables -- the 42/217 partial scores are dropped and the
# spearman cells are left empty with this status. The pretrain loss (the LR-grid result,
# received in §2) is kept, since it is a real measurement and is what marks the model
# as non-learning.
CUTOFF = {("bert", "xl")}
CUTOFF_NOTE = "未測定・事前学習で非学習のため打ち切り"


def bert_loss_step(run_dir):
    cks = sorted((int(os.path.basename(p).split("-")[-1]) for p in glob.glob(str(run_dir / "checkpoint-*"))), reverse=True)
    for ck in cks:  # newest checkpoint that has a trainer_state (a live run may be mid-write)
        f = run_dir / f"checkpoint-{ck}" / "trainer_state.json"
        if f.exists():
            ts = json.load(open(f))
            e = [x["eval_loss_mask"] for x in ts.get("log_history", []) if "eval_loss_mask" in x]
            return (min(e) if e else None), ck
    return None, None


def gpt2_loss_step(run_dir):
    best = (1e9, None)
    for cf in sorted(glob.glob(str(run_dir / "logging_*.csv"))):
        for row in csv.reader(open(cf)):
            if len(row) < 3:
                continue
            try:
                it = int(row[0])
                vl = float(row[2])
            except ValueError:
                continue
            if vl < best[0]:
                best = (vl, it)
    return (best[0] if best[1] is not None else None), best[1]


def load_reference():
    ref = {}
    if not REF.exists():
        return ref
    with open(REF) as fh:
        for row in csv.DictReader(fh):
            ref[row["DMS_id"]] = row
    return ref


def assay_spearman_n(slug, assay):
    mf = EVAL / slug / f"proteingym_{assay}_sweep" / "metrics.json"
    if not mf.exists():
        return None, None
    try:
        d = json.load(open(mf))
        s = d["metrics"].get("spearman")
        n = d.get("details", {}).get("num_variants")
        if s is None or math.isnan(float(s)):
            return None, n
        return float(s), n
    except Exception:
        return None, None


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/protein_proteingym_tables.py <out_dir>")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    assays = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(PGDIR / "*.csv")))
    ref = load_reference()

    # precompute per-model loss/step/status
    meta = {}
    for arch, size, lr, seed, slug, predir, lossmetric, kind in MODELS:
        loss, step = (bert_loss_step(predir) if kind == "bert" else gpt2_loss_step(predir))
        n_done = sum(1 for a in assays if (EVAL / slug / f"proteingym_{a}_sweep" / "metrics.json").exists())
        if not (EVAL / slug).exists() or n_done == 0:
            status = "not_started"
        elif n_done >= len(assays):
            status = "done"
        else:
            status = "running"
        meta[(arch, size)] = dict(lr=lr, seed=seed, slug=slug, loss=loss, step=step,
                                  lossmetric=lossmetric, status=status, n_done=n_done)

    # ---- §1 per-assay ----
    p1 = out / "proteingym-per-assay.tsv"
    with open(p1, "w") as fh:
        fh.write("arch\tsize\tlr\tseed\tcheckpoint\tassay\tn_mutant\tspearman\tstatus\n")
        for arch, size, lr, seed, slug, _predir, _lossmetric, _kind in MODELS:
            m = meta[(arch, size)]
            ck = m["step"] if m["step"] is not None else ""
            for a in assays:
                nmut = ref.get(a, {}).get("DMS_total_number_mutants", "")
                if (arch, size) in CUTOFF:
                    fh.write(f"{arch}\t{size}\t{lr}\t{seed}\t{ck}\t{a}\t{nmut}\t\t{CUTOFF_NOTE}\n")
                    continue
                sp, _n = assay_spearman_n(slug, a)
                if sp is not None:
                    fh.write(f"{arch}\t{size}\t{lr}\t{seed}\t{ck}\t{a}\t{nmut}\t{sp:.6f}\tdone\n")
                else:
                    st = "not_started" if m["status"] == "not_started" else "running"
                    fh.write(f"{arch}\t{size}\t{lr}\t{seed}\t{ck}\t{a}\t{nmut}\t\t{st}\n")

    # ---- §2 per-model ----
    p2 = out / "proteingym-per-model.tsv"
    with open(p2, "w") as fh:
        fh.write("arch\tsize\tlr\tcheckpoint\tpretrain_loss\tloss_metric\tn_assay_done\tspearman_mean\tspearman_median\tspearman_min\tspearman_max\tstatus\n")
        for arch, size, lr, _seed, slug, _predir, lossmetric, _kind in MODELS:
            m = meta[(arch, size)]
            ck = m["step"] if m["step"] is not None else ""
            loss = f"{m['loss']:.4f}" if m["loss"] is not None else ""
            if (arch, size) in CUTOFF:
                # keep the real pretrain loss; drop every ProteinGym cell.
                fh.write(f"{arch}\t{size}\t{lr}\t{ck}\t{loss}\t{lossmetric}\t\t\t\t\t\t{CUTOFF_NOTE}\n")
                continue
            vals = []
            for a in assays:
                sp, _ = assay_spearman_n(slug, a)
                if sp is not None:
                    vals.append(sp)
            if vals:
                mean = f"{statistics.mean(vals):.4f}"
                med = f"{statistics.median(vals):.4f}"
                lo = f"{min(vals):.4f}"
                hi = f"{max(vals):.4f}"
            else:
                mean = med = lo = hi = ""
            st = "done" if m["status"] == "done" else ("not_started" if m["status"] == "not_started" else "running")
            fh.write(f"{arch}\t{size}\t{lr}\t{ck}\t{loss}\t{lossmetric}\t{len(vals)}\t{mean}\t{med}\t{lo}\t{hi}\t{st}\n")

    # ---- §3 set stats ----
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"assays (CSV list): {len(assays)}; reference rows: {len(ref)}")
    nm = [int(ref[a]["DMS_total_number_mutants"]) for a in assays if a in ref and ref[a].get("DMS_total_number_mutants", "").isdigit()]
    orgs = {ref[a].get("source_organism", "") for a in assays if a in ref}
    sel = {}
    for a in assays:
        if a in ref:
            sel[ref[a].get("selection_type", "")] = sel.get(ref[a].get("selection_type", ""), 0) + 1
    if nm:
        print(f"n_mutant range: {min(nm)}..{max(nm)}  median {int(statistics.median(nm))}")
    print(f"distinct source_organism: {len(orgs)}")
    print(f"selection_type counts: {sel}")
    print("per-model status:")
    for arch, size, *_ in MODELS:
        m = meta[(arch, size)]
        print(f"  {arch} {size}: status={m['status']} n_done={m['n_done']} loss={m['loss']} step={m['step']}")


if __name__ == "__main__":
    main()
