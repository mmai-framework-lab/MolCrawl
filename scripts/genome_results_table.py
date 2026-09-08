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
    ap.add_argument("--seq-len", type=int, default=None,
                    help="fallback only; the run manifest is preferred")
    ap.add_argument("--global-batch", type=int, default=None,
                    help="fallback only; the run manifest is preferred")
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

        # Prefer what the run recorded over what this script was told. The
        # manifest carries tokens_per_step already, computed from the batch the
        # run actually used, so a CLI default cannot disagree with it.
        batch = man.get("batch") or {}
        per_step = batch.get("tokens_per_step")
        if per_step is None:
            gb = batch.get("effective_global_batch") or args.global_batch
            sl = batch.get("seq_len") or args.seq_len
            per_step = (gb * sl) if (gb and sl) else None
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
            "training_tokens": (steps * per_step) if (steps and per_step) else None,
            "tokens_per_step": per_step,
            "window_loss_fraction": w.get("loss_fraction"),
            # Segments, not lines. A raw line is a wrapped piece of an N-split
            # segment, so the line-level share is capped by the 261,120 wrap and
            # understates how much of a chromosome-scale assembly yields nothing.
            "no_window_segment_fraction": w.get("no_window_segment_fraction"),
            "mean_segment_len": w.get("mean_segment_len"),
        })

    rows.sort(key=lambda r: (-(r["margin"] if r["margin"] is not None else 0)))

    def cell(v, fmt):
        """A covariate that is missing shows as missing.

        A subset whose manifest or window-loss entry has not been produced yet
        must not take the whole table down with it -- the scores are the point,
        and a blank cell says which input is absent far better than a traceback.
        """
        return "-" if v is None else fmt(v)

    print("| subset | eval_loss_mask | 散らばり | 基準線 | 余裕 | 余裕の散らばり |"
          " step | 学習トークン | 窓欠落 | 窓0のsegment |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["subset"],
            cell(r["eval_loss_mask"], lambda v: f"{v:.4f}"),
            cell(r["eval_loss_mask_spread"], lambda v: f"{v:.4f}"),
            cell(r["degenerate_baseline"], lambda v: f"{v:.4f}"),
            cell(r["margin"], lambda v: f"**{v:+.4f}**"),
            cell(r["margin_spread"], lambda v: f"{v:.4f}"),
            cell(r["max_steps"], lambda v: f"{v:,}"),
            cell(r["training_tokens"], lambda v: f"{v/1e9:.1f}B"),
            cell(r["window_loss_fraction"], lambda v: f"{100*v:.2f}%"),
            cell(r["no_window_segment_fraction"], lambda v: f"{100*v:.1f}%"),
        ))

    missing = [r["subset"] for r in rows
               if r["max_steps"] is None or r["window_loss_fraction"] is None]
    if missing:
        print(f"\n  ⚠️  共変量が欠けている subset: {', '.join(missing)}")

    if any(r["margin"] is not None for r in rows):
        m = [r["margin"] for r in rows if r["margin"] is not None]
        sp = max((r["margin_spread"] for r in rows
                  if r["margin_spread"] is not None), default=0)
        print(f"\n  余裕の幅 {min(m):+.4f} 〜 {max(m):+.4f} = {max(m)-min(m):.4f}")
        print(f"  マスク draw による最大の散らばり {sp:.4f}")
        print(f"  → subset 間の差は draw の {(max(m)-min(m))/sp:.1f} 倍" if sp else "")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
