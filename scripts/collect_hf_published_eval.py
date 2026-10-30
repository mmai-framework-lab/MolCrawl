"""Collect the training-eval series of published models from a Hugging Face org.

Each published repo carries its own ``trainer_state.json``, so the whole eval
history of a run survives even when the machine that produced it is gone. This
reads them and writes one row per run, so a published model can be judged without
retraining or re-scoring it.

What the numbers can and cannot say is fixed by what was recorded: the June
generation logged only ``eval_loss``, the 80/10/10 blend. A blended value cannot
decide whether a run sits on its degenerate floor -- the true [MASK]-only loss is
somewhere between ``eval_loss`` and ``eval_loss / 0.9``. Both bounds are reported
so the ambiguity is visible rather than hidden behind one number.

    python scripts/collect_hf_published_eval.py --author <org> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MODALITY = [
    ("protein", r"protein"),
    ("genome", r"genome"),
    ("rna", r"\brna\b|scrna"),
    ("molecule_nat_lang", r"mol.?instruct|nat.?lang|molnl"),
    ("compounds", r"compound|chembl|guacamol|zinc|pubchem"),
]
ARCH = [("bert", r"\bbert\b"), ("roberta", r"roberta"), ("gpt2", r"gpt-?2"), ("llama", r"llama"), ("esm2", r"esm2")]
SIZE = [("ex-large", r"ex-?large|xl\b"), ("large", r"large"), ("medium", r"medium"), ("small", r"small")]


def classify(name: str, table, default="?"):
    low = name.lower()
    for label, pat in table:
        if re.search(pat, low):
            return label
    return default


def series(state: dict, key: str):
    pts = [(e["step"], e[key]) for e in state.get("log_history", []) if key in e]
    pts.sort()
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--author", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--filter", default="", help="only repos whose id contains this")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    api = HfApi()
    models = sorted(m for m in (x.id for x in api.list_models(author=args.author)) if args.filter in m)
    print(f"{len(models)} repos under {args.author}")

    rows, missing = [], []
    for i, repo in enumerate(models, 1):
        try:
            p = hf_hub_download(repo, "trainer_state.json")
        except (EntryNotFoundError, RepositoryNotFoundError, OSError) as e:
            missing.append((repo, type(e).__name__))
            print(f"  [{i:2d}/{len(models)}] {repo}  -- no trainer_state.json")
            continue
        st = json.loads(Path(p).read_text())
        ev = series(st, "eval_loss")
        mask = series(st, "eval_loss_mask")
        if not ev:
            missing.append((repo, "no eval_loss in log_history"))
            print(f"  [{i:2d}/{len(models)}] {repo}  -- no eval points")
            continue
        steps = [s for s, _ in ev]
        vals = [v for _, v in ev]
        vmin = min(vals)
        half = len(vals) // 2 or 1
        # How much of the total descent happened in the second half of training.
        drop_all = vals[0] - vmin
        drop_late = min(vals[:half]) - vmin
        rows.append(
            {
                "repo": repo,
                "modality": classify(repo, MODALITY),
                "arch": classify(repo, ARCH),
                "size": classify(repo, SIZE),
                "eval_points": len(ev),
                "max_step": steps[-1],
                "epochs": round(st.get("epoch") or 0, 2),
                "eval_loss_final": round(vals[-1], 4),
                "eval_loss_min": round(vmin, 4),
                "min_at_step": steps[vals.index(vmin)],
                "mask_only_lower": round(vals[-1], 4),
                "mask_only_upper": round(vals[-1] / 0.9, 4),
                "late_share_of_descent": round(drop_late / drop_all, 3) if drop_all > 0 else 0.0,
                "has_eval_loss_mask": bool(mask),
            }
        )
        print(f"  [{i:2d}/{len(models)}] {repo}  {len(ev)} pts  final {vals[-1]:.4f}  min {vmin:.4f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "hf-published-eval.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "hf-published-eval.json").write_text(
        json.dumps({"author": args.author, "rows": rows, "missing": missing}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out / 'hf-published-eval.csv'} ({len(rows)} rows), {len(missing)} without a usable log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
