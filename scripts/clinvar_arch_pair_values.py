"""The AUROCs behind each architecture comparison, one row per subset.

A median and a range say which side won and by how much in the middle. They do
not say what either side scored, and they cannot be checked against anything.
This writes the two values the difference was taken from, for every subset and
every row of the comparison, with the series named on each row.

The pairing is preserved: a row holds one subset's BERT value and the same
subset's GPT-2 value, which is what the difference was computed from. Taking a
mean of one column and a mean of the other would answer a different question.

Values are read from where the campaign wrote them -- probe.json for the linear
probe, each subset's metrics.json for zero-shot -- rather than recomputed, so
the table and the figures cannot disagree.
"""
import argparse
import csv
import glob
import json
import os

FOLDS = ("chr21", "chr22", "chrX")


def probe_values(path):
    """{subset: {"overall": v, "chr21": v, ...}} from a probe.json."""
    runs = json.load(open(path))["runs"]
    out = {}
    for subset, got in runs.items():
        row = {"overall": got["overall_auroc"]}
        for fold, v in got.get("per_fold", {}).items():
            row[fold] = v["auroc"]
        out[subset] = row
    return out


def zeroshot_values(root):
    """{subset: {"overall": v}} from each subset's metrics.json.

    Zero-shot writes no per-fold AUROC; the fold-wise numbers come from a
    separate analysis over predictions.jsonl, so only the overall value belongs
    in this table.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
        subset = os.path.basename(os.path.dirname(path))
        out[subset] = {"overall": json.load(open(path))["metrics"]["auroc"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    R = args.results_root
    probe_bert = probe_values(f"{R}/clinvar-probe-bert-w1026-v2/probe.json")
    probe_gpt2_9 = probe_values(f"{R}/clinvar-probe-gpt2-ep9/probe.json")
    probe_gpt2_3 = probe_values(f"{R}/clinvar-probe-gpt2/probe.json")
    zs_bert = zeroshot_values(f"{R}/clinvar-zeroshot-bert-w1026")
    zs_gpt2_9 = zeroshot_values(f"{R}/clinvar-zeroshot-gpt2-ep9")

    # The six rows of the 2026-10-15 report's §7 table, in its order.
    comparisons = [
        ("線形プローブ", "overall", "bert-w1026", probe_bert, "gpt2-ep9", probe_gpt2_9),
        ("線形プローブ", "chr21", "bert-w1026", probe_bert, "gpt2-ep9", probe_gpt2_9),
        ("線形プローブ", "chr22", "bert-w1026", probe_bert, "gpt2-ep9", probe_gpt2_9),
        ("線形プローブ", "chrX", "bert-w1026", probe_bert, "gpt2-ep9", probe_gpt2_9),
        ("ゼロショット", "overall", "bert-w1026", zs_bert, "gpt2-ep9", zs_gpt2_9),
        ("線形プローブ", "overall", "bert-w1026", probe_bert, "gpt2", probe_gpt2_3),
    ]

    rows = []
    for method, scope, b_series, b_vals, g_series, g_vals in comparisons:
        for subset in sorted(set(b_vals) & set(g_vals)):
            b, g = b_vals[subset].get(scope), g_vals[subset].get(scope)
            if b is None or g is None:
                continue
            rows.append({"method": method, "scope": scope, "subset": subset,
                         "bert_series": b_series, "bert_auroc": f"{b:.6f}",
                         "gpt2_series": g_series, "gpt2_auroc": f"{g:.6f}",
                         "diff_gpt2_minus_bert": f"{g - b:+.6f}",
                         "favours": "BERT" if g < b else "GPT-2"})

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    with open(args.out_tsv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {args.out_tsv}  ({len(rows):,} rows)")

    for method, scope, b_series, _b, g_series, _g in comparisons:
        got = [r for r in rows if r["method"] == method and r["scope"] == scope
               and r["gpt2_series"] == g_series]
        n_bert = sum(1 for r in got if r["favours"] == "BERT")
        print(f"  {method} {scope:8s} {b_series} 対 {g_series:9s} "
              f"n={len(got):>2d}  BERT 有利 {n_bert}/{len(got)}")

    if args.out_md:
        lines = ["# アーキテクチャ比較の元になった AUROC", ""]
        for method, scope, b_series, _b, g_series, _g in comparisons:
            got = [r for r in rows if r["method"] == method and r["scope"] == scope
                   and r["gpt2_series"] == g_series]
            lines += [f"## {method} / {scope} / {b_series} 対 {g_series}", "",
                      f"| subset | BERT（{b_series}） | GPT-2（{g_series}） | 差 GPT-2 − BERT |",
                      "|---|---|---|---|"]
            for r in got:
                lines.append(f"| {r['subset']} | {float(r['bert_auroc']):.4f} | "
                             f"{float(r['gpt2_auroc']):.4f} | {r['diff_gpt2_minus_bert']} |")
            lines.append("")
        open(args.out_md, "w").write("\n".join(lines))
        print(f"  wrote {args.out_md}")


if __name__ == "__main__":
    main()
