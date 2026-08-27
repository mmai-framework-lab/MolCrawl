"""Pick the checkpoint to adopt from a finished MLM run, by `eval_loss_mask`.

The run selects its own "best" with `metric_for_best_model`, which is `eval_loss` -
the blend over every scored position, including the copy positions that a working
model gets almost free. Adoption is on `[MASK]` positions only, so the run's own
output is not necessarily the one to take.

Reads the per-checkpoint JSON that scripts/eval_mlm_checkpoint.py writes and reports
which step wins, with the full table, so the adopted step is shown to be the minimum
rather than asserted.
"""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path


def main() -> int:
    parser = ArgumentParser(description="Adopt the checkpoint with the lowest [MASK] loss")
    parser.add_argument("--score-dir", required=True, help="dir of eval_mlm_checkpoint.py outputs")
    parser.add_argument("--label", required=True, help="label prefix the scores were written under")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scores = []
    for path in sorted(Path(args.score_dir).glob(f"{args.label}_step*.json")):
        payload = json.loads(path.read_text())
        step = int(path.stem.rsplit("step", 1)[1])
        scores.append((step, payload["loss_mask"], payload.get("loss_copy"), payload.get("eval_subset"), str(path)))
    if not scores:
        raise SystemExit(f"no scores matching {args.label}_step*.json under {args.score_dir}")

    scores.sort()
    best = min(scores, key=lambda r: r[1])

    print(f"=== {args.label}: adoption by eval_loss_mask ===")
    print(f"{'step':>8s} {'[MASK]':>10s} {'copy':>10s}   ")
    for step, mask, copy, _, _ in scores:
        mark = "  <- adopted (minimum)" if step == best[0] else ""
        print(f"{step:8,} {mask:10.4f} {copy:10.4f}{mark}")
    margin = sorted(r[1] for r in scores)
    gap = margin[1] - margin[0] if len(margin) > 1 else None
    print(f"\nadopted step {best[0]:,}, eval_loss_mask {best[1]:.4f}"
          + (f", next best is {gap:+.4f} away" if gap is not None else ""))
    print(f"checkpoints scored: {len(scores)}  eval slice: {scores[0][3]}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "label": args.label,
            "adopted_step": best[0],
            "adopted_loss_mask": best[1],
            "runner_up_gap": gap,
            "eval_subset": best[3],
            "scored": [{"step": s, "loss_mask": m, "loss_copy": c} for s, m, c, _, _ in scores],
        }, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
