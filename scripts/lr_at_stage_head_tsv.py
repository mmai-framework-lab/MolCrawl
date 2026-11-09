"""The learning rate HF actually logged at the head of a continued stage.

A continuation here is launched from a COPY of the previous stage's checkpoint with
``scheduler.pt`` removed, so the scheduler is built fresh from the config. With
``lr_scheduler_type = "constant_with_warmup"`` and ``warmup_steps = 500`` a fresh
scheduler starts its ramp at zero, which means the rate drops to nothing at the
resume point and climbs back over 500 steps -- even where the previous stage ended
at a constant rate rather than at zero.

That is a claim about the scheduler's state after a resume, and the configs say it
both ways: the 30,000-step config's comment reads "there is no jump from nothing at
this resume point as there was at 15,000", which is about the rate the PREVIOUS
stage ended on, not about where the new ramp begins. Reading the scheduler's code to
settle it would still be reading. ``log_history`` carries the rate Trainer logged at
every ``logging_steps``, so the question is answered by the run.

    lr_at_stage_head_tsv.py --run DIR@HEAD_STEP [--run ...] [--window 600] --out FILE

HEAD_STEP is the global step the stage resumed from. Rows cover that step and the
window after it, plus the last rate the run logged.

Read-only: ``log_history`` out of the furthest checkpoint. Paths are arguments; this
repository is public.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = ["run", "stage_head", "step", "offset_from_head", "learning_rate",
           "kind", "source"]


def split_run(spec: str) -> tuple[Path, int]:
    """``DIR@HEAD_STEP``. The head step cannot be read out of the run."""
    if "@" not in spec:
        raise SystemExit(f"--run needs DIR@HEAD_STEP, got {spec!r}")
    head, _, tail = spec.rpartition("@")
    return Path(head), int(tail)


def newest_trainer_state(run: Path) -> Path:
    steps = sorted(int(p.name.split("-")[1]) for p in run.glob("checkpoint-*") if p.is_dir())
    if not steps:
        raise SystemExit(f"no checkpoint-* directory under {run}")
    path = run / f"checkpoint-{steps[-1]}" / "trainer_state.json"
    if not path.exists():
        raise SystemExit(f"no trainer_state.json in {path.parent}")
    return path


def rate_series(log_history) -> dict[int, float]:
    """{step: learning_rate}. A step logged twice keeps the last value."""
    out = {}
    for entry in log_history:
        if "learning_rate" in entry and entry.get("step") is not None:
            out[int(entry["step"])] = float(entry["learning_rate"])
    return out


def rows_for(run: Path, head: int, window: int) -> list[dict]:
    state_path = newest_trainer_state(run)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rates = rate_series(state.get("log_history", []))
    if not rates:
        raise SystemExit(f"no learning_rate in {state_path}")

    source = f"{run.name}/{state_path.parent.name}/{state_path.name}"
    last = max(rates)
    wanted = [s for s in sorted(rates) if head <= s <= head + window]
    if last not in wanted:
        wanted.append(last)

    out = []
    for step in wanted:
        if step == head:
            kind = "head (previous stage)"
        elif step == last and step > head + window:
            kind = "last logged"
        elif step <= head + 500:
            kind = "inside the 500-step ramp"
        else:
            kind = "after the ramp"
        out.append({
            "run": run.name,
            "stage_head": head,
            "step": step,
            "offset_from_head": step - head,
            "learning_rate": f"{rates[step]:.6e}",
            "kind": kind,
            "source": source,
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, metavar="DIR@HEAD_STEP",
                    help="a run directory and the step it resumed from; repeat")
    ap.add_argument("--window", type=int, default=600,
                    help="how many steps after the head to list (default 600)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    for spec in args.run:
        run, head = split_run(spec)
        rows.extend(rows_for(run, head, args.window))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"  {row['run'][:46]:46s} step {row['step']:>6,} "
              f"(+{row['offset_from_head']:>4,})  lr {row['learning_rate']}  {row['kind']}")
    print(f"\n{len(rows)} row -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
