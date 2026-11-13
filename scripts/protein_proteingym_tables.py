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
EVAL = REPO / "learning_source_20260730_protein_uniref50/experiment_data/eval"
PGDIR = REPO / "learning_source_20260730_protein_uniref50/eval/proteingym/unpacked/DMS_ProteinGym_substitutions"
REF = REPO / "learning_source_20260625-newBERT/protein_sequence/proteingym/proteingym_v1.3/DMS_substitutions.csv"

# arch, size, lr, train seed, eval slug, pretrain-loss source dir, loss metric, kind
#
# protein-order-2026-11-02 §3 (decision B) + protein-judgment-2026-11-09 §5 +
# protein-judgment-2026-11-10 §3.2/§5: every model is now sourced from its seed-42
# confirmed series and scored on that same series' ADOPTED checkpoint (the …-s42 slugs).
# The GPT-2 four moved off their seed-1001 runs onto the seed-42 -rerun runs, BERT large
# moved off the production run onto -rerun, and BERT small/medium were rescored on their
# argmin eval_loss_mask checkpoint (= best_model_checkpoint: 32000 / 32900) instead of the
# final 33531 -- so every model's reached loss and its downstream score now come from the
# same run and the same step. BERT xl is the cut-off model.
MODELS = [
    ("bert", "small", "3e-4", 42, "protein_sequence-bert-small-s42", RUNS_PROD / "bert-output/protein_sequence-small-lr3e4", "eval_loss_mask", "bert"),
    ("bert", "medium", "1e-4", 42, "protein_sequence-bert-medium-s42", RUNS_PROD / "bert-output/protein_sequence-medium-lr1e4", "eval_loss_mask", "bert"),
    ("bert", "large", "3.5e-5", 42, "protein_sequence-bert-large-s42", RUNS_PROD / "bert-output/protein_sequence-large-lr3p5e5-rerun", "eval_loss_mask", "bert"),
    ("bert", "xl", "2.4e-5", 42, "protein_sequence-bert-xl-best", RUNS_PROD / "bert-output/protein_sequence-xl-lr2p4e5", "eval_loss_mask", "bert"),
    ("gpt2", "small", "1.2e-3", 42, "protein_sequence-gpt2-small-s42", RUNS_PROD / "gpt2-output/protein_sequence-small-rerun", "val_loss", "gpt2"),
    ("gpt2", "medium", "6e-4", 42, "protein_sequence-gpt2-medium-s42", RUNS_PROD / "gpt2-output/protein_sequence-medium-rerun", "val_loss", "gpt2"),
    ("gpt2", "large", "6e-4", 42, "protein_sequence-gpt2-large-s42", RUNS_PROD / "gpt2-output/protein_sequence-large-rerun", "val_loss", "gpt2"),
    ("gpt2", "xl", "6e-4", 42, "protein_sequence-gpt2-xl-s42", RUNS_PROD / "gpt2-output/protein_sequence-ex-large-rerun", "val_loss", "gpt2"),
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
            # The step the loss column reports is where eval_loss_mask is lowest, not the
            # newest checkpoint dir: for BERT small/medium the newest saved dir (33531) has
            # no eval logged, and the run's best is earlier (32000 / 32900). This argmin step
            # equals the run's best_model_checkpoint, which is the checkpoint now scored.
            e = [(x["step"], x["eval_loss_mask"]) for x in ts.get("log_history", []) if "eval_loss_mask" in x]
            if not e:
                return None, None
            step, val = min(e, key=lambda t: t[1])
            return val, step
    return None, None


def bert_best_ckpt(run_dir):
    """The run's best_model_checkpoint step, read from the newest trainer_state.json.

    protein-judgment-2026-11-10 §3.3: this is the checkpoint HF Trainer marked best, which
    the reached-loss table checks the scored checkpoint against. None if unrecorded.
    """
    cks = sorted((int(os.path.basename(p).split("-")[-1]) for p in glob.glob(str(run_dir / "checkpoint-*"))), reverse=True)
    for ck in cks:
        f = run_dir / f"checkpoint-{ck}" / "trainer_state.json"
        if f.exists():
            bmc = json.load(open(f)).get("best_model_checkpoint")
            if bmc:
                return int(os.path.basename(str(bmc).rstrip("/")).split("-")[-1])
            return None
    return None


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
        # scored_ck: the step of the checkpoint ProteinGym actually ran on, read from a
        # scored metrics.json. It sits alongside spearman in the per-assay/per-model tables,
        # so it names the downstream weights -- for GPT-2 the ckpt.pt (its adopted step) and
        # for BERT the scored checkpoint dir (now the adopted checkpoint: 32000 small/large,
        # 32900 medium). It equals the best-loss step (meta["step"]); the reached-loss table
        # joins the two and checks them against best_model_checkpoint. Empty when unscored.
        scored_ck = ""
        for a in assays:
            mf = EVAL / slug / f"proteingym_{a}_sweep" / "metrics.json"
            if mf.exists():
                try:
                    base = os.path.basename(json.load(open(mf))["details"]["model_path"].rstrip("/"))
                    scored_ck = step if kind == "gpt2" else (int(base.split("-")[-1]) if base.startswith("checkpoint-") else "")
                except Exception:
                    scored_ck = ""
                break
        meta[(arch, size)] = dict(lr=lr, seed=seed, slug=slug, loss=loss, step=step,
                                  lossmetric=lossmetric, status=status, n_done=n_done, scored_ck=scored_ck)

    # ---- §1 per-assay ----
    p1 = out / "proteingym-per-assay.tsv"
    with open(p1, "w") as fh:
        fh.write("arch\tsize\tlr\tseed\tcheckpoint\tassay\tn_mutant\tspearman\tstatus\n")
        for arch, size, lr, seed, slug, _predir, _lossmetric, _kind in MODELS:
            m = meta[(arch, size)]
            ck = "" if (arch, size) in CUTOFF else m["scored_ck"]
            for a in assays:
                nmut = ref.get(a, {}).get("DMS_total_number_mutants", "")
                if (arch, size) in CUTOFF:
                    fh.write(f"{arch}\t{size}\t{lr}\t{seed}\t{ck}\t{a}\t{nmut}\t\t{CUTOFF_NOTE}\n")
                    continue
                sp, _n = assay_spearman_n(slug, a)
                if sp is not None:
                    fh.write(f"{arch}\t{size}\t{lr}\t{seed}\t{ck}\t{a}\t{nmut}\t{sp:.6f}\tdone\n")
                else:
                    # metrics.json present but spearman undefined (e.g. tied predictions) is
                    # "nan", not "running": the assay was scored, it just has no usable value.
                    mf = EVAL / slug / f"proteingym_{a}_sweep" / "metrics.json"
                    if mf.exists():
                        st = "nan"
                    else:
                        st = "not_started" if m["status"] == "not_started" else "running"
                    fh.write(f"{arch}\t{size}\t{lr}\t{seed}\t{ck}\t{a}\t{nmut}\t\t{st}\n")

    # ---- §2 per-model ----
    p2 = out / "proteingym-per-model.tsv"
    with open(p2, "w") as fh:
        # n_assay_scored = assays with a complete metrics.json (the sweep ran); n_assay_nonnan
        # = those with a defined spearman (the denominator of the mean). The gap is the handful
        # of assays whose spearman is undefined for every model -- 217 vs 214 here.
        fh.write("arch\tsize\tlr\tcheckpoint\tpretrain_loss\tloss_metric\tn_assay_scored\tn_assay_nonnan\tspearman_mean\tspearman_median\tspearman_min\tspearman_max\tstatus\n")
        for arch, size, lr, _seed, slug, _predir, lossmetric, _kind in MODELS:
            m = meta[(arch, size)]
            ck = "" if (arch, size) in CUTOFF else m["scored_ck"]
            loss = f"{m['loss']:.4f}" if m["loss"] is not None else ""
            if (arch, size) in CUTOFF:
                # keep the real pretrain loss; drop every ProteinGym cell.
                fh.write(f"{arch}\t{size}\t{lr}\t{ck}\t{loss}\t{lossmetric}\t\t\t\t\t\t\t{CUTOFF_NOTE}\n")
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
            fh.write(f"{arch}\t{size}\t{lr}\t{ck}\t{loss}\t{lossmetric}\t{m['n_done']}\t{len(vals)}\t{mean}\t{med}\t{lo}\t{hi}\t{st}\n")

    # ---- reached pretraining loss, one row per model (protein-judgment-2026-11-09 §5,
    #      -2026-11-10 §3.3) ----
    # Each model's reached loss and the checkpoint actually scored for ProteinGym, so loss
    # and downstream can be traced to the same run and the same step. For GPT-2 the scored
    # file is ckpt.pt (nanoGPT keeps the best-val checkpoint), whose step is not in the
    # filename -- it is in loss_step; GPT-2's ckpt.pt is best by construction, so it is out
    # of scope for the best_model_checkpoint check (§3.3). For pretrained BERT the scored
    # checkpoint dir is checked against the run's best_model_checkpoint, and the note states
    # the result one line each.
    def scored_checkpoint(slug):
        for a in assays:
            mf = EVAL / slug / f"proteingym_{a}_sweep" / "metrics.json"
            if mf.exists():
                try:
                    mp = json.load(open(mf))["details"]["model_path"]
                except Exception:
                    return ""
                return os.path.basename(mp.rstrip("/"))
        return ""

    p4 = out / "proteingym-reached-loss.tsv"
    with open(p4, "w") as fh:
        fh.write("arch\tsize\tlr\trun\tseed\tloss_metric\treached_loss\tloss_step\t"
                 "scored_checkpoint\tbest_model_checkpoint\tscored_eq_best\tvalidation_set\tnote\n")
        for arch, size, lr, seed, slug, predir, lossmetric, _kind in MODELS:
            m = meta[(arch, size)]
            run = os.path.basename(str(predir))
            loss = f"{m['loss']:.4f}" if m["loss"] is not None else ""
            step = m["step"] if m["step"] is not None else ""
            valset = "run val split (seed 42)"
            best_col = ""      # best_model_checkpoint step (BERT only)
            eq_col = ""        # yes / no / — ; whether the scored checkpoint is the best one
            if (arch, size) in CUTOFF:
                sc = "—（下流打ち切り）"
                best_col = "—"
                eq_col = "—"
                note = CUTOFF_NOTE
            elif _kind == "bert":
                sc = scored_checkpoint(slug)
                best = bert_best_ckpt(predir)
                best_col = f"checkpoint-{best}" if best is not None else "(not recorded)"
                sc_step = sc.split("-")[-1] if sc.startswith("checkpoint-") else ""
                if best is not None and sc_step == str(best):
                    eq_col = "yes"
                    note = "scored checkpoint matches best_model_checkpoint"
                elif best is not None:
                    eq_col = "no"
                    note = (f"scored {sc} != best_model_checkpoint checkpoint-{best} "
                            "(verify unless the latest was adopted on purpose)")
                else:
                    eq_col = "—"
                    note = "best_model_checkpoint not recorded"
            else:  # gpt2: ckpt.pt is best by construction -- out of scope for the §3.3 check
                sc = scored_checkpoint(slug)
                best_col = "n/a (GPT-2 ckpt.pt, out of scope)"
                eq_col = "—"
                note = ""
            fh.write(f"{arch}\t{size}\t{lr}\t{run}\t{seed}\t{lossmetric}\t{loss}\t{step}\t"
                     f"{sc}\t{best_col}\t{eq_col}\t{valset}\t{note}\n")

    # ---- §3 set stats ----
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"wrote {p4}")
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
