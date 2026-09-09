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
import json
import os
import sys

import numpy as np

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


def representations(rows, model_path, tokenizer_path, arch, batch_size, device):
    """Frozen features: what the substitution did to the representation.

    The hidden state at the variant position, before and after the substitution,
    and the difference -- plus the same difference mean-pooled over the window,
    since a substitution can move the representation of positions around it.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    centre = len(rows[0]["reference_sequence"]) // 2

    def encode(seqs):
        ids = [tok.convert_tokens_to_ids(list(s)) for s in seqs]
        if arch != "gpt2":
            cls, sep = tok.cls_token_id, tok.sep_token_id
            ids = [[cls] + i + [sep] for i in ids]
        return torch.tensor(ids, dtype=torch.long, device=device)

    offset = 0 if arch == "gpt2" else 1        # [CLS] shifts every position by one
    feats = []
    with torch.no_grad():
        for lo in range(0, len(rows), batch_size):
            chunk = rows[lo:lo + batch_size]
            h_ref = model(input_ids=encode([r["reference_sequence"] for r in chunk])
                          ).last_hidden_state
            h_var = model(input_ids=encode([r["variant_sequence"] for r in chunk])
                          ).last_hidden_state
            at = centre + offset
            d_centre = h_var[:, at] - h_ref[:, at]
            d_mean = h_var.mean(dim=1) - h_ref.mean(dim=1)
            feats.append(torch.cat([h_ref[:, at], d_centre, d_mean], dim=1).float().cpu().numpy())
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


def probe(features, rows, C, seed):
    """Fit on the other chromosomes, score the held-out one. Three times."""
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
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        clf.fit(sc.transform(features[tr]), y[tr])
        s = clf.predict_proba(sc.transform(features[te]))[:, 1]
        held[te] = s
        per_fold["chr" + test[0]] = {
            "n_test": int(te.sum()), "n_train": int(tr.sum()),
            "pathogenic_test": int(y[te].sum()), "auroc": auroc(y[te], s),
        }
    return per_fold, auroc(y, held), held


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
    ap.add_argument("--seed", type=int, default=1026)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model-free-only", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import glob
    import torch

    chroms = {c.strip() for c in args.chroms.split(",") if c.strip()}
    rows = read_table(args.clinvar_csv, chroms)
    y = np.array([r["_y"] for r in rows])
    print(f"  variants {len(rows):,}   pathogenic {int(y.sum()):,}   "
          f"benign {int((y == 0).sum()):,}   C={args.C}  seed={args.seed}")
    print(f"  folds: test chr21 / chr22 / chrX, chr{'/'.join(ALWAYS_TRAIN)} always in train\n")

    results = {"C": args.C, "seed": args.seed, "variants": len(rows), "runs": {}}

    mf = model_free_features(rows)
    per_fold, overall, _ = probe(mf, rows, args.C, args.seed)
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
        for d in sorted(glob.glob(os.path.join(args.runs_root, args.run_glob))):
            run = os.path.basename(d.rstrip("/"))
            subset = run.split("-small-", 1)[-1]
            if args.run_tag:
                subset = subset[: -len(args.run_tag) - 1] if subset.endswith(
                    "-" + args.run_tag) else subset
            ck = [int(p.rsplit("-", 1)[1]) for p in glob.glob(os.path.join(d, "checkpoint-*"))
                  if p.rsplit("-", 1)[1].isdigit()]
            if not ck:
                print(f"  {subset}: checkpoint なし")
                continue
            state = json.load(open(os.path.join(d, f"checkpoint-{max(ck)}",
                                                "trainer_state.json")))
            best = state.get("best_model_checkpoint")
            step = int(os.path.basename(best).rsplit("-", 1)[1]) if best else max(ck)
            path = os.path.join(d, f"checkpoint-{step}")

            feats = representations(rows, path, args.tokenizer, args.arch,
                                    args.batch_size, device)
            pf, ov, _ = probe(feats, rows, args.C, args.seed)
            results["runs"][subset] = {"checkpoint": f"checkpoint-{step}",
                                       "overall_auroc": ov, "per_fold": pf,
                                       "n_features": int(feats.shape[1])}
            print(f"  {subset:44s} 全体 {ov:.4f}   "
                  + "  ".join(f"{k} {v['auroc']:.4f}" for k, v in pf.items()))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
