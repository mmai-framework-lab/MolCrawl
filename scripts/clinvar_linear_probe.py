"""Linear probe on frozen representations, against a model-free control.

Fine-tuning trains the trunk, so a weak pretrained representation can be made up
for by the head and the comparison stops being about pretraining. Freezing the
trunk and fitting one logistic regression removes that: with the features fixed
and the regularisation fixed, the fit is determined, and any difference between
subsets is a difference in what the pretraining left behind.

Splits rotate over chromosomes, never over variants. Variants of one gene sit
within a few hundred bases of each other and the windows are 1,024 wide, so a
variant-level split puts overlapping sequence on both sides -- measured at 67% of
held-out variants having a training variant within 128 bases on the old build.
Rotating chromosomes cannot overlap at all.

chrY is always in the training side: 29 variants is not a test set.

The model-free control runs on the same folds, from features no model produces --
the substitution type, whether the site is CpG, the GC content of the window. If
a subset's probe does not beat that, its pretraining has added nothing a lookup
table does not already have.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

from molcrawl.models._representations import build_encoder, special_token_offset

FOLDS = [("21",), ("22",), ("X",)]     # chrY stays in train in all three
ALWAYS_TRAIN = ("Y",)
BASES = "ACGT"


def read_table(path, chroms):
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows = []
    for r in csv.DictReader(open(path, newline="")):
        c = (r.get("chrom") or "").replace("chr", "").strip()
        if c not in chroms:
            continue
        sig = (r.get("ClinicalSignificance") or "").lower()
        if "conflicting" in sig:
            continue
        p, b = "pathogenic" in sig, "benign" in sig
        if p == b:
            continue
        r["_chrom"], r["_y"] = c, 1 if p else 0
        rows.append(r)
    return rows


def model_free_features(rows):
    """What a lookup table knows: the substitution, CpG, and GC content."""
    subs = [(a, b) for a in BASES for b in BASES if a != b]      # 12
    index = {s: i for i, s in enumerate(subs)}
    out = np.zeros((len(rows), len(subs) + 2), dtype=np.float32)
    for i, r in enumerate(rows):
        ref, alt = r["ref"].upper(), r["alt"].upper()
        if (ref, alt) in index:
            out[i, index[(ref, alt)]] = 1.0
        seq = r["reference_sequence"]
        mid = len(seq) // 2
        # CpG at the site: the substituted base with either neighbour
        cpg = (seq[mid:mid + 2] == "CG") or (seq[max(0, mid - 1):mid + 1] == "CG")
        out[i, len(subs)] = 1.0 if cpg else 0.0
        out[i, len(subs) + 1] = sum(c in "GC" for c in seq) / max(1, len(seq))
    return out


def variant_token_index(arch, window_len):
    """Where the variant sits in the token sequence.

    The window puts the variant at its centre; the offset is whatever special
    token the family prepends. Reading the wrong index scores a position that is
    not the variant, and nothing in the output would say so.
    """
    return window_len // 2 + special_token_offset(arch)


def adopted_checkpoint(run_dir, arch):
    """The checkpoint the run adopted -- not its newest one.

    nanoGPT rewrites ckpt.pt at the run root whenever validation improves, so
    that file is the best-val model. HF keeps every checkpoint and names the
    adopted one in the newest trainer_state; on the 21 genome BERT runs the
    adopted and newest checkpoints differ in every one.
    """
    if arch == "gpt2":
        path = os.path.join(run_dir, "ckpt.pt")
        return (path, "ckpt.pt") if os.path.exists(path) else (None, None)
    steps = [int(p.rsplit("-", 1)[1]) for p in glob.glob(os.path.join(run_dir, "checkpoint-*"))
             if p.rsplit("-", 1)[1].isdigit()]
    if not steps:
        return None, None
    state = json.load(open(os.path.join(run_dir, f"checkpoint-{max(steps)}",
                                        "trainer_state.json")))
    best = state.get("best_model_checkpoint")
    step = int(os.path.basename(best).rsplit("-", 1)[1]) if best else max(steps)
    path = os.path.join(run_dir, f"checkpoint-{step}")
    return (path, f"checkpoint-{step}") if os.path.isdir(path) else (None, None)


def representations(rows, forward, encode, at, batch_size):
    """Frozen features: what the substitution did to the representation.

    The hidden state at the variant position, the change there, and the same
    change averaged over the window, since a substitution moves the positions
    around it too.

    In a causal model the positions before the variant are identical between the
    two sequences, so half the window contributes exactly zero to the third
    group and its mean is half the mean over the right-hand side. That is a
    constant factor on the whole group, and the per-fold standardisation removes
    it.
    """
    import torch

    feats = []
    with torch.no_grad():
        for lo in range(0, len(rows), batch_size):
            chunk = rows[lo:lo + batch_size]
            h_ref = forward(encode([r["reference_sequence"] for r in chunk]))
            h_var = forward(encode([r["variant_sequence"] for r in chunk]))
            d_centre = h_var[:, at] - h_ref[:, at]
            d_mean = h_var.mean(dim=1) - h_ref.mean(dim=1)
            feats.append(torch.cat([h_ref[:, at], d_centre, d_mean],
                                   dim=1).float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def probe(features, rows, C, seed, max_iter=20000):
    """Fit on the other chromosomes, score the held-out one. Three times.

    ``max_iter`` is a stopping limit, not a setting that shapes the answer: a fit
    that reaches its tolerance first is unaffected by raising it. It has to be
    generous because untrained representations are far harder to separate than
    trained ones -- at 2,000 the trained runs and the model-free control all
    converged while every untrained fit hit the limit, which would have left the
    floor lower than it really is and made pretraining look better than it is.
    Whether each fit converged is recorded rather than left to a warning on
    stderr.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    chrom = np.array([r["_chrom"] for r in rows])
    y = np.array([r["_y"] for r in rows])
    per_fold, held = {}, np.full(len(rows), np.nan)
    for test in FOLDS:
        te = np.isin(chrom, test)
        tr = ~te                              # includes chrY by construction
        # The scaler is fitted on the training side only; fitting it on
        # everything would let the held-out chromosome inform the transform.
        sc = StandardScaler().fit(features[tr])
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        clf.fit(sc.transform(features[tr]), y[tr])
        s = clf.predict_proba(sc.transform(features[te]))[:, 1]
        held[te] = s
        n_iter = int(np.asarray(clf.n_iter_).max())
        per_fold["chr" + test[0]] = {
            "n_test": int(te.sum()), "n_train": int(tr.sum()),
            "pathogenic_test": int(y[te].sum()), "auroc": auroc(y[te], s),
            "n_iter": n_iter, "converged": bool(n_iter < max_iter),
        }
    # chrY is never held out, so its entries stay NaN. np.argsort sorts NaN
    # last, which would rank those 29 variants as the most pathogenic; the
    # overall value is taken over the variants that were actually scored.
    ok = ~np.isnan(held)
    return per_fold, auroc(y[ok], held[ok]), held


def write_predictions(path, rows, held):
    """One line per held-out variant: what the paired comparisons are built from.

    Without these, the fold AUROCs are all that survives and nothing can be
    compared variant for variant afterwards -- neither probe against probe nor
    probe against the model-free control. chrY is never held out and is omitted.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for r, s in zip(rows, held):
            if np.isnan(s):
                continue
            fh.write(json.dumps({"vcv_id": r.get("vcv_id"), "chrom": r["_chrom"],
                                 "fold": "chr" + r["_chrom"],
                                 "label_pathogenic": int(r["_y"]),
                                 "score": float(s)}) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar-csv", required=True)
    ap.add_argument("--runs-root", default="")
    ap.add_argument("--run-glob", default="")
    ap.add_argument("--arch", default="bert")
    ap.add_argument("--tokenizer", default="",
                    help="required unless --model-free-only")
    ap.add_argument("--run-tag", default="w1026")
    ap.add_argument("--chroms", default="21,22,X,Y")
    ap.add_argument("--C", type=float, default=1.0,
                    help="inverse regularisation; one value for every run and fold")
    ap.add_argument("--max-iter", type=int, default=20000,
                    help="stopping limit for the fit, not a tuning knob; a fit "
                         "that reaches tolerance first is unaffected")
    ap.add_argument("--seed", type=int, default=1026)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model-free-only", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--untrained-seeds", default="",
                    help="comma-separated seeds; builds the architecture from a "
                         "run's config and leaves the weights at init, instead "
                         "of probing the trained runs")
    ap.add_argument("--pred-dir", default="",
                    help="write <pred-dir>/<name>/predictions.jsonl for each probe")
    args = ap.parse_args()

    import torch

    chroms = {c.strip() for c in args.chroms.split(",") if c.strip()}
    rows = read_table(args.clinvar_csv, chroms)
    y = np.array([r["_y"] for r in rows])
    print(f"  variants {len(rows):,}   pathogenic {int(y.sum()):,}   "
          f"benign {int((y == 0).sum()):,}   C={args.C}  seed={args.seed}  "
          f"max_iter={args.max_iter}")
    print(f"  folds: test chr21 / chr22 / chrX, chr{'/'.join(ALWAYS_TRAIN)} always in train\n")

    # The fit is described in the output rather than left to the reader to infer
    # from defaults: the same settings have to hold for every run and every fold,
    # and a later sklearn could change what a default means.
    from sklearn.linear_model import LogisticRegression as _LR
    _ref = _LR(C=args.C, max_iter=args.max_iter)
    results = {"C": args.C, "seed": args.seed, "max_iter": args.max_iter,
               "penalty": _ref.penalty, "solver": _ref.solver,
               "standardisation": "StandardScaler, per feature, fitted on each "
                                  "fold's training side only",
               "variants": len(rows), "runs": {},
               # Features are taken in fp32; the runs trained under bf16 autocast.
               # Every subset goes through this same path, so the comparison
               # across subsets is unaffected.
               "precision": "fp32",
               "feature_layout": "h_ref[centre] | h_var[centre]-h_ref[centre] | "
                                 "mean(h_var)-mean(h_ref)"}

    mf = model_free_features(rows)
    per_fold, overall, held = probe(mf, rows, args.C, args.seed, args.max_iter)
    if args.pred_dir:
        write_predictions(os.path.join(args.pred_dir, "model_free", "predictions.jsonl"),
                          rows, held)
    results["model_free"] = {"overall_auroc": overall, "per_fold": per_fold,
                             "n_features": int(mf.shape[1])}
    print(f"  {'model-free (置換型 12 + CpG + GC)':44s} 全体 {overall:.4f}   "
          + "  ".join(f"{k} {v['auroc']:.4f}" for k, v in per_fold.items()))

    if not args.model_free_only and args.runs_root and not args.tokenizer:
        raise SystemExit("--tokenizer is required when scoring model representations")

    if args.model_free_only or not args.runs_root:
        print("\n  (モデル側は未実行)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        runs = sorted(glob.glob(os.path.join(args.runs_root, args.run_glob)))
        seeds = [int(x) for x in args.untrained_seeds.split(",") if x.strip()]
        jobs = []                       # (name, checkpoint, label, untrained, seed)
        if seeds:
            # One architecture, several initialisations. The reference run gives
            # the shape only -- its weights are never read -- so which run it is
            # does not matter, but it is recorded.
            ref = next((d.rstrip("/") for d in runs
                        if adopted_checkpoint(d.rstrip("/"), args.arch)[0]), None)
            if ref is None:
                raise SystemExit(f"no run under {args.run_glob} carries a checkpoint "
                                 f"to take the architecture from")
            path, label = adopted_checkpoint(ref, args.arch)
            for sd in seeds:
                jobs.append((f"untrained_seed{sd}", path,
                             f"{os.path.basename(ref)}/{label} の構成のみ", True, sd))
        else:
            for d in runs:
                d = d.rstrip("/")
                subset = os.path.basename(d).split("-small-", 1)[-1]
                if args.run_tag and subset.endswith("-" + args.run_tag):
                    subset = subset[: -len(args.run_tag) - 1]
                path, label = adopted_checkpoint(d, args.arch)
                if path is None:
                    print(f"  {subset}: checkpoint なし")
                    continue
                jobs.append((subset, path, label, False, 0))

        at = variant_token_index(args.arch, len(rows[0]["reference_sequence"]))
        results["variant_token_index"] = at
        print(f"  arch {args.arch}   変異のトークン位置 {at}\n")

        for name, path, label, untrained, sd in jobs:
            t0 = time.monotonic()
            fwd, enc, hidden = build_encoder(path, args.tokenizer, args.arch, device,
                                             untrained=untrained, init_seed=sd)
            feats = representations(rows, fwd, enc, at, args.batch_size)
            t_feat = time.monotonic() - t0
            pf, ov, held = probe(feats, rows, args.C, args.seed, args.max_iter)
            t_all = time.monotonic() - t0
            if args.pred_dir:
                write_predictions(os.path.join(args.pred_dir, name, "predictions.jsonl"),
                                  rows, held)
            results["runs"][name] = {"checkpoint": label,
                                     "untrained": untrained,
                                     "init_seed": sd if untrained else None,
                                     "hidden_size": hidden,
                                     "overall_auroc": ov, "per_fold": pf,
                                     "n_features": int(feats.shape[1]),
                                     "seconds_features": round(t_feat, 1),
                                     "seconds_total": round(t_all, 1)}
            print(f"  {name:36s} 全体 {ov:.4f}   "
                  + "  ".join(f"{k} {v['auroc']:.4f}" for k, v in pf.items())
                  + f"   ({label}, hidden {hidden}, {t_all / 60:.1f} 分)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
