"""Assemble the 21-run results table, with the covariates it must be read beside.

The verdict of 2026-09-07 asks that window loss travel with the results rather
than be corrected out of them: loss rate follows from the contig length
distribution, which follows from how the subset was drawn, so it is an
independent variable of the experiment and belongs in the same table as the
numbers it moves.

Step count differs per subset by design. Fixing the epoch count fixes how many
times each run sees its own data; it does not fix compute or tokens, and the
learning rate decays over each run's own max_steps, so every run finishes its own
schedule on its own data.
"""
import argparse
import glob
import json
import os


def _load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True, help="per-subset json from scoring")
    ap.add_argument("--window-loss", required=True, help="window-loss.json")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--run-tag", default="w1026")
    ap.add_argument("--seq-len", type=int, default=1026)
    ap.add_argument("--global-batch", type=int, default=2560)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    wl = {r["subset"]: r for r in (_load(args.window_loss) or [])}
    rows = []
    for f in sorted(glob.glob(os.path.join(args.scores_dir, "*.json"))):
        s = os.path.splitext(os.path.basename(f))[0]
        sc = _load(f)
        if not sc:
            continue
        run = os.path.join(args.runs_root,
                           f"bert-small-{s}" + (f"-{args.run_tag}" if args.run_tag else ""))
        man = _load(os.path.join(run, "run_manifest.json")) or {}
        steps = (man.get("schedule") or {}).get("max_steps")
        w = wl.get(s, {})
        rows.append({
            "subset": s,
            "eval_loss_mask": sc["eval_loss_mask"]["mean"],
            "eval_loss_mask_spread": sc["eval_loss_mask"]["spread"],
            "degenerate_baseline": sc["degenerate_baseline"]["mean"],
            "margin": sc["margin"]["mean"],
            "margin_spread": sc["margin"]["spread"],
            "checkpoint": os.path.basename(sc.get("checkpoint", "")),
            "max_steps": steps,
            # Tokens the run was trained on: sequences x length. Not the same as
            # bases, since every sequence carries [CLS] and [SEP].
            "training_tokens": (steps * args.global_batch * args.seq_len) if steps else None,
            "window_loss_fraction": w.get("loss_fraction"),
            "zero_window_contig_fraction": w.get("short_contig_fraction"),
            "mean_contig_len": w.get("mean_contig_len"),
        })

    rows.sort(key=lambda r: (-(r["margin"] or 0)))
    hdr = ("| subset | eval_loss_mask | 散らばり | 基準線 | 余裕 | 余裕の散らばり |"
           " step | 学習トークン | 窓欠落 | 窓0のcontig |")
    print(hdr)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| `{r['subset']}` | {r['eval_loss_mask']:.4f} | {r['eval_loss_mask_spread']:.4f} |"
              f" {r['degenerate_baseline']:.4f} | **{r['margin']:+.4f}** | {r['margin_spread']:.4f} |"
              f" {r['max_steps']:,} | {r['training_tokens']/1e9:.1f}B |"
              f" {100*r['window_loss_fraction']:.2f}% | {100*r['zero_window_contig_fraction']:.1f}% |")

    if rows:
        m = [r["margin"] for r in rows]
        sp = max(r["margin_spread"] for r in rows)
        print(f"\n  余裕の幅 {min(m):+.4f} 〜 {max(m):+.4f} = {max(m)-min(m):.4f}")
        print(f"  マスク draw による最大の散らばり {sp:.4f}")
        print(f"  → subset 間の差は draw の {(max(m)-min(m))/sp:.1f} 倍" if sp else "")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
