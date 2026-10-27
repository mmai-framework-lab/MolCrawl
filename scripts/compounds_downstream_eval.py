"""Downstream evaluation of the compounds models on frozen representations.

Fine-tuning trains the trunk, so a weak pretrained representation can be made up for
by the head and the comparison stops being about pretraining. Freezing the trunk and
fitting one linear model removes that: with the features fixed and the regularisation
fixed, the fit is determined, and a difference between models is a difference in what
pretraining left behind. Same design as the genome ClinVar probe
(`scripts/clinvar_linear_probe.py`).

Two tasks, and they are not the same kind:

  Solubility_AqSolDB      9,982   regression, reported as RMSE
  herg_central/hERG_inhib 306,893 binary, reported as ROC-AUC

Splits rotate over Bemis-Murcko scaffolds, never over molecules. Analogues of one
scaffold would otherwise sit on both sides and the numbers would read as better than
the models are. Three folds, each held out once, so every molecule is scored exactly
once and the spread between folds is a spread over different held-out chemistry.

`moleculenet.splits.scaffold_split` is **not** used for the rotation: measured
2026-10-16, its `seed` only shuffles within the three sets it produces, so calling it
three times returns the same held-out molecules. Its scaffold function is reused; the
assignment to folds is done here, greedily placing the largest group into the
currently smallest fold, which is deterministic and needs no seed.

**The checkpoint used is the one the run adopted** -- `best_model_checkpoint` from the
newest `trainer_state` for BERT, `ckpt.pt` for nanoGPT -- not the minimum of the whole
log history. The two differ when the minimum was never saved, which is the case for
compounds BERT medium: its `eval_loss_mask` minimum at step 14,700 has no checkpoint
and the adopted one is step 15,000. Reporting the adopted checkpoint's own value
alongside the result is what keeps the two from being confused.

Two controls, because a number with no floor under it says nothing:

  ECFP4 + gradient boosting   a fingerprint no model produced
  the same architecture untrained   the representation before pretraining
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np



def read_tdc(path, smiles_col, label_col):
    """SMILES and labels out of one of TDC's .tab files."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            smi = (r.get(smiles_col) or "").strip().strip('"')
            raw = r.get(label_col)
            if not smi or raw in (None, ""):
                continue
            try:
                y = float(raw)
            except ValueError:
                continue
            rows.append({"smiles": smi, "y": y})
    return rows


def adopted_checkpoint(run_dir, arch):
    """The checkpoint the run adopted, not its newest one.

    nanoGPT rewrites ckpt.pt whenever validation improves, so that file is the
    best-val model. HF names the adopted one in the newest trainer_state.
    """
    if arch == "gpt2":
        p = os.path.join(run_dir, "ckpt.pt")
        return (p, "ckpt.pt") if os.path.exists(p) else (None, None)
    steps = [int(p.rsplit("-", 1)[1]) for p in glob.glob(os.path.join(run_dir, "checkpoint-*"))
             if p.rsplit("-", 1)[1].isdigit()]
    if not steps:
        return None, None
    with open(os.path.join(run_dir, f"checkpoint-{max(steps)}", "trainer_state.json"),
              encoding="utf-8") as fh:
        state = json.load(fh)
    best = state.get("best_model_checkpoint")
    step = int(os.path.basename(best).rsplit("-", 1)[1]) if best else max(steps)
    p = os.path.join(run_dir, f"checkpoint-{step}")
    return (p, f"checkpoint-{step}") if os.path.isdir(p) else (None, None)


def adopted_value(run_dir, arch):
    """What the adopted checkpoint scored, so it is not confused with the minimum."""
    if arch == "gpt2":
        return None
    steps = [int(p.rsplit("-", 1)[1]) for p in glob.glob(os.path.join(run_dir, "checkpoint-*"))
             if p.rsplit("-", 1)[1].isdigit()]
    if not steps:
        return None
    with open(os.path.join(run_dir, f"checkpoint-{max(steps)}", "trainer_state.json"),
              encoding="utf-8") as fh:
        state = json.load(fh)
    return {"best_metric": state.get("best_metric"),
            "best_model_checkpoint": os.path.basename(str(state.get("best_model_checkpoint") or ""))}


def scaffold_folds(smiles, n_folds=3):
    """Rotate scaffold groups over `n_folds` folds; each is held out once.

    Groups are assigned largest-first to whichever fold is smallest so far, which
    balances the folds without a random draw. A molecule whose scaffold cannot be
    computed becomes its own group, so it is never grouped with an unrelated one.
    """
    sys.path.insert(0, os.getcwd())
    from collections import defaultdict

    from molcrawl.tasks.evaluation.moleculenet.splits import _bemis_murcko_scaffold

    groups = defaultdict(list)
    for i, smi in enumerate(smiles):
        key = _bemis_murcko_scaffold(smi) or f"__unparsed_{i}"
        groups[key].append(i)

    assigned = [[] for _ in range(n_folds)]
    for members in sorted(groups.values(), key=lambda b: (-len(b), b[0])):
        target = min(range(n_folds), key=lambda f: (len(assigned[f]), f))
        assigned[target].extend(members)

    folds = []
    for f in range(n_folds):
        test = np.asarray(sorted(assigned[f]), dtype=int)
        train = np.asarray(sorted(i for g in range(n_folds) if g != f
                                  for i in assigned[g]), dtype=int)
        folds.append((train, test))
    return folds, len(groups)


def ecfp4(smiles, n_bits=2048):
    """Morgan fingerprints, radius 2 -- ECFP4 -- as the model-free control."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[i] = np.asarray(gen.GetFingerprint(mol), dtype=np.float32)
    return out


def tokenize_smiles(smiles, tokenizer_path, max_length, arch):
    """Token ids in the tokenisation the compounds models were trained on.

    ``build_encoder``'s own ``encode`` splits with ``list(s)``. That is right for a
    character-level family and wrong here: 579 of the 612 entries in the compounds
    vocabulary are multi-character SMILES tokens, so ``list()`` turns ``Br`` into
    ``B`` + ``r``, ``Cl`` into ``C`` + ``l`` and ``[O-]`` into five characters.
    Measured on ``CC(=O)Oc1ccccc1C(=O)[O-].[Na+]``: 23 tokens through the lexer, 30
    by character. Feeding a model a tokenisation it was never trained on is the same
    class of error as dropping position_ids, which inflated a reported [MASK] loss
    about six-fold.

    HF trunks get [CLS] and [SEP] the way ``_hf_encoder`` adds them; nanoGPT gets
    neither, matching its own encode. The returned ``content`` slice says which
    positions are the molecule, so pooling never averages a special token in.
    """
    sys.path.insert(0, os.getcwd())
    from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer

    tok = CompoundsTokenizer(tokenizer_path, max_length)
    vocab = tok.vocab if hasattr(tok, "vocab") else None
    unk = (vocab or {}).get("[UNK]")
    cls_id = (vocab or {}).get("[CLS]")
    sep_id = (vocab or {}).get("[SEP]")
    if arch != "gpt2" and (cls_id is None or sep_id is None):
        raise SystemExit("the tokenizer has no [CLS]/[SEP]; cannot match _hf_encoder")

    out, skipped = [], 0
    budget = max_length - (2 if arch != "gpt2" else 0)
    for s in smiles:
        pieces = tok.tokenize(s)
        if len(pieces) > budget or not pieces:
            skipped += 1
            out.append(None)
            continue
        ids = [(vocab or {}).get(t, unk) for t in pieces]
        if any(i is None for i in ids):
            skipped += 1
            out.append(None)
            continue
        if arch == "gpt2":
            out.append((ids, 0, len(ids)))
        else:
            out.append(([cls_id] + ids + [sep_id], 1, 1 + len(ids)))
    return out, skipped


def representations(smiles, model_path, arch, tokenizer_path, device,
                    max_length=1024, untrained=False, init_seed=0, batch_rows=64):
    """Mean-pooled final hidden states, one row per molecule.

    ``build_encoder`` supplies ``forward`` and the hidden size, so the layer and the
    final-normalisation convention are the shared ones -- after ``ln_f`` for nanoGPT,
    ``last_hidden_state`` for HF.

    **No padding anywhere.** That module pads nothing and builds no attention mask,
    so a padded position would be attended to and would change the hidden states of
    the real tokens by an amount that varies with how much padding a molecule
    happens to need. Sequences are therefore grouped by exact token length and every
    batch is uniform, which is the contract that module states.
    """
    import numpy as np
    import torch

    sys.path.insert(0, os.getcwd())
    from molcrawl.models._representations import build_encoder

    encoded, skipped = tokenize_smiles(smiles, tokenizer_path, max_length, arch)
    # No tokenizer_path: this function tokenises with CompoundsTokenizer, because
    # build_encoder's own encode splits with list(s) and 579 of the 612 compounds
    # tokens are multi-character. Passing the vocab.txt path would fail anyway --
    # AutoTokenizer wants a directory holding a tokenizer config, and compounds has
    # none. Only forward and the hidden size are taken from here.
    forward, _encode_unused, hidden = build_encoder(
        model_path, None, arch, device,
        untrained=untrained, init_seed=init_seed)

    feats = np.zeros((len(smiles), hidden), dtype=np.float32)
    usable = [i for i, e in enumerate(encoded) if e is not None]
    by_len = {}
    for i in usable:
        by_len.setdefault(len(encoded[i][0]), []).append(i)

    with torch.no_grad():
        for length in sorted(by_len):
            idxs = by_len[length]
            for off in range(0, len(idxs), batch_rows):
                chunk = idxs[off:off + batch_rows]
                ids = torch.tensor([encoded[i][0] for i in chunk],
                                   dtype=torch.long, device=device)
                h = forward(ids)
                lo, hi = encoded[chunk[0]][1], encoded[chunk[0]][2]
                pooled = h[:, lo:hi].mean(dim=1).float().cpu().numpy()
                for row, i in zip(pooled, chunk):
                    feats[i] = row
    return feats, hidden, skipped, len(usable)


def fit_and_score(features, y, folds, task, seed=42, max_iter=20000):
    """Fit on the training side of each fold, score the held-out side."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    per_fold = []
    for k, (tr, te) in enumerate(folds):
        # The scaler is fitted on the training side only: fitting it on everything
        # would let the held-out scaffolds inform the transform.
        sc = StandardScaler().fit(features[tr])
        xt, xe = sc.transform(features[tr]), sc.transform(features[te])
        if task == "classification":
            clf = LogisticRegression(C=1.0, max_iter=max_iter, random_state=seed)
            clf.fit(xt, y[tr])
            s = clf.predict_proba(xe)[:, 1]
            score = roc_auc(y[te], s)
            n_iter = int(np.asarray(clf.n_iter_).max())
            conv = n_iter < max_iter
        else:
            clf = Ridge(alpha=1.0, random_state=seed)
            clf.fit(xt, y[tr])
            s = clf.predict(xe)
            score = float(np.sqrt(np.mean((y[te] - s) ** 2)))
            n_iter, conv = None, True
        per_fold.append({"fold": k, "n_train": int(tr.size), "n_test": int(te.size),
                         "score": score, "n_iter": n_iter, "converged": bool(conv)})
    vals = [f["score"] for f in per_fold]
    return per_fold, float(np.mean(vals)), float(np.std(vals))


def roc_auc(labels, scores):
    """Rank-based AUROC; ties share their average rank."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average the ranks of equal scores, or a model that outputs one value scores 1.0
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=("solubility", "herg"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--control-only", action="store_true",
                    help="only the ECFP4 control; no model is loaded")
    ap.add_argument("--model", action="append", default=[], metavar="ARCH:SIZE:DIR",
                    help="a run to read representations from; repeat for several")
    ap.add_argument("--tokenizer", default="assets/molecules/vocab.txt",
                    help="the SMILES vocabulary the runs were trained with")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--untrained-seeds", type=int, default=0,
                    help="how many untrained trunks to score as a floor, per arch")
    ap.add_argument("--limit", type=int, default=None, help="for a smoke test")
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    smiles_col, label_col, kind = (("Drug", "Y", "regression") if args.task == "solubility"
                                   else ("X", "hERG_inhib", "classification"))
    rows = read_tdc(args.data, smiles_col, label_col)
    if args.limit:
        rows = rows[: args.limit]
    smiles = [r["smiles"] for r in rows]
    y = np.asarray([r["y"] for r in rows], dtype=float)
    print(f"  {args.task}: {len(rows):,} 分子  課題 {kind}  "
          + (f"陽性 {int(y.sum()):,}" if kind == "classification" else
             f"範囲 [{y.min():.3f}, {y.max():.3f}]"), flush=True)

    t0 = time.time()
    folds, n_groups = scaffold_folds(smiles, n_folds=3)
    for k, (tr, te) in enumerate(folds):
        print(f"    分割 {k}: train {tr.size:,} / test {te.size:,}", flush=True)
    covered = sum(te.size for _, te in folds)
    print(f"    骨格 {n_groups:,} 群 / 全分子が一度ずつ test: "
          f"{covered == len(rows)}  ({time.time() - t0:.0f} 秒)", flush=True)

    results = {"task": args.task, "metric": "RMSE" if kind == "regression" else "ROC-AUC",
               "molecules": len(rows), "split": "Bemis-Murcko scaffold groups rotated over 3 folds",
               "scaffold_groups": n_groups, "models": {}}

    print("  --- ECFP4 対照 ---", flush=True)
    t0 = time.time()
    fp = ecfp4(smiles)
    per_fold, mean, sd = fit_and_score(fp, y, folds, kind)
    results["models"]["ECFP4+linear"] = {"arch": "none", "per_fold": per_fold,
                                         "mean": mean, "sd": sd}
    print(f"    ECFP4: {mean:.4f} ± {sd:.4f}  ({time.time() - t0:.0f} 秒)", flush=True)

    for spec in (args.model if not args.control_only else []):
        arch, size, run_dir = spec.split(":", 2)
        ckpt, ckpt_name = adopted_checkpoint(run_dir, arch)
        if ckpt is None:
            print(f"    {arch} {size}: 採用 checkpoint が無い（{run_dir}）", flush=True)
            continue
        print(f"  --- {arch} {size}  {ckpt_name} ---", flush=True)
        t0 = time.time()
        feats, hidden, skipped, used = representations(
            smiles, ckpt, arch, args.tokenizer, args.device,
            max_length=args.max_length)
        print(f"    表現 {used:,} 分子 × {hidden}  （長さ超過などで除外 {skipped:,}）"
              f"  ({time.time() - t0:.0f} 秒)", flush=True)
        per_fold, mean, sd = fit_and_score(feats, y, folds, kind)
        results["models"][f"{arch}-{size}"] = {
            "arch": arch, "size": size, "run_dir": run_dir,
            "adopted_checkpoint": ckpt_name,
            "adopted_value": adopted_value(run_dir, arch),
            "hidden_size": hidden, "skipped": skipped, "scored": used,
            "per_fold": per_fold, "mean": mean, "sd": sd,
        }
        print(f"    {arch} {size}: {mean:.4f} ± {sd:.4f}", flush=True)

        for s in range(args.untrained_seeds):
            feats_u, _h, _sk, _us = representations(
                smiles, ckpt, arch, args.tokenizer, args.device,
                max_length=args.max_length, untrained=True, init_seed=s)
            pf, m, d = fit_and_score(feats_u, y, folds, kind)
            results["models"][f"{arch}-{size}-untrained-seed{s}"] = {
                "arch": arch, "size": size, "untrained": True, "init_seed": s,
                "per_fold": pf, "mean": m, "sd": d}
            print(f"    {arch} {size} 未学習 seed{s}: {m:.4f} ± {d:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n記録 {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
