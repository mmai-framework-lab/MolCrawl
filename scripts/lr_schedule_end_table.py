"""How each modality's learning rate ends, read from the runs and their configs.

Written because the claim "every modality decays to zero at the end" was made
without being checked. It is true of one architecture and false of the other, and
the difference is in the framework rather than in any modality's config:

  BERT   The HF Trainer's default ``lr_scheduler_type`` is "linear", which decays
         from the peak to zero over ``max_steps``. A config that leaves the key
         unset therefore ends at zero. The rate the last step ran at is logged, so
         it is read from ``log_history`` rather than derived.
  GPT-2  nanoGPT's ``get_lr`` is cosine from ``learning_rate`` down to ``min_lr``
         and holds ``min_lr`` past ``lr_decay_iters``. ``min_lr`` is peak/10 by
         convention in this repository, so these runs end at a tenth of the peak,
         not at zero. nanoGPT logs the rate only to TensorBoard, so it is derived
         from the config here -- and the config values are read as text, because
         several of these config modules load a dataset at import time.

A config inherited through ``from ... import *`` is followed, with the child's
value winning, so a two-line override file still reports the rate it runs at.

``schedule_end_lr`` is where the schedule ends; ``last_logged_lr`` is the last rate
the log happens to carry. They differ for a reason worth keeping apart: logging lands
on an interval, so a linear schedule that truly reaches zero at ``max_steps`` still
shows 1e-7 on its last line, and reading that line would say "not zero".

Read-only. Every path is an argument: this repository is public.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

KEYS = ("learning_rate", "min_lr", "max_iters", "lr_decay_iters", "warmup_iters",
        "decay_lr", "max_steps", "warmup_steps", "lr_scheduler_type")
ASSIGN = re.compile(r"^(%s)\s*=\s*(.+?)\s*(?:#.*)?$" % "|".join(KEYS), re.M)
STAR_IMPORT = re.compile(
    r"^from\s+(molcrawl\.tasks\.pretrain\.configs\.[\w.]+)\s+import\s+\*", re.M)
COLUMNS = ["modality", "arch", "run", "steps_run", "schedule", "peak_lr",
           "schedule_end_lr", "ends_at_zero", "last_logged_lr", "last_logged_step",
           "config", "config_lines", "source"]


def config_values(path: Path, repo: Path, _seen=None) -> dict:
    """The lr-shaping assignments in a config, with its star-imported parent's first.

    Values are kept as the text on the right of the ``=`` and the line number, so
    that ``1.2e-3 / 10`` is reported as written rather than silently evaluated.
    """
    _seen = _seen or set()
    if path in _seen or not path.exists():
        return {}
    _seen.add(path)
    text = path.read_text(encoding="utf-8")
    out = {}
    for parent in STAR_IMPORT.findall(text):
        out.update(config_values(repo / (parent.replace(".", "/") + ".py"), repo, _seen))
    for key, value in ASSIGN.findall(text):
        line = next(i for i, src in enumerate(text.splitlines(), 1)
                    if re.match(rf"^{key}\s*=", src))
        out[key] = (value, f"{path.name}:{line}")
    return out


ENV_DEFAULT = re.compile(r'os\.environ\.get\(\s*"[^"]+"\s*,\s*"([^"]+)"\s*\)')


def as_number(expr: str):
    """``1.2e-3 / 10``, ``learning_rate / 10`` and an env override are all written here.

    An ``os.environ.get(KEY, "0.0006")`` is read as its literal default, which is
    what a launch that does not set KEY uses; genome's subset configs take the rate
    that way. A launch that does set it is out of reach of a text read, and the
    config line is in the table so that can be checked.
    """
    expr = ENV_DEFAULT.sub(r"\1", expr)
    # float and int are in scope because the configs wrap the env read in float();
    # nothing else is, so an expression naming anything further returns None.
    try:
        return float(eval(expr, {"__builtins__": {}},  # noqa: S307 - digits only
                          {"float": float, "int": int}))
    except Exception:
        return None


def newest_checkpoint(run: Path) -> Path | None:
    cks = [p for p in run.glob("checkpoint-*") if (p / "trainer_state.json").exists()]
    return max(cks, key=lambda p: int(p.name.split("-")[1])) if cks else None


def bert_row(modality: str, run: Path, config: Path, repo: Path) -> dict:
    """The rate the last logged step ran at, taken from the run."""
    checkpoint = newest_checkpoint(run)
    if checkpoint is None:
        return {"modality": modality, "arch": "bert", "run": run.name,
                "schedule": "no checkpoint with a trainer_state"}
    with open(checkpoint / "trainer_state.json") as fh:
        state = json.load(fh)
    rates = [(h["step"], h["learning_rate"]) for h in state["log_history"]
             if "learning_rate" in h]
    cfg = config_values(config, repo) if config else {}
    kind = cfg.get("lr_scheduler_type", ("None (HF default: linear)", ""))[0]
    step, final = rates[-1] if rates else ("", "")
    peak = max((r for _, r in rates), default=None)
    # The endpoint of the schedule, not the last line of the log. Logging lands on an
    # interval, so a linear schedule's last logged rate is a tenth of a percent of the
    # peak rather than zero even when the schedule itself reaches zero at max_steps.
    if kind.startswith("None"):
        end, zero = 0.0, "yes"           # HF "linear": peak -> 0 over max_steps
    elif "constant" in kind:
        end = as_number(cfg.get("learning_rate", ("", ""))[0])
        zero = "no" if end else ""
    else:
        end, zero = None, ""
    return {
        "modality": modality, "arch": "bert", "run": run.name,
        "steps_run": state.get("global_step", ""),
        "schedule": kind.strip('"'),
        "peak_lr": f"{peak:.3e}" if peak is not None else "",
        "schedule_end_lr": f"{end:.3e}" if end is not None else "",
        "ends_at_zero": zero,
        "last_logged_lr": f"{final:.3e}" if rates else "",
        "last_logged_step": step,
        "config": config.name if config else "",
        "config_lines": "; ".join(v[1] for k, v in sorted(cfg.items())
                                  if k in ("lr_scheduler_type", "max_steps",
                                           "warmup_steps", "learning_rate")),
        "source": f"{run.name}/{checkpoint.name}/trainer_state.json"
                  f" (last of {len(rates)} logged rates, step {step})",
    }


def nanogpt_last_iter(run: Path) -> str:
    """The last iteration the run actually logged, from its own csv.

    The config cannot supply it: the subset configs derive ``max_iters`` from the
    row count of a dataset they load at import time.
    """
    csvs = sorted(run.glob("logging_*.csv"))
    if not csvs:
        return ""
    last = ""
    with open(csvs[-1], encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            head = line.split(",", 1)[0].strip()
            if head.isdigit():
                last = head
    return last


def gpt2_row(modality: str, run: Path, config: Path, repo: Path) -> dict:
    """The rate the last iteration ran at, derived from get_lr's resting value.

    Past ``lr_decay_iters`` nanoGPT returns ``min_lr`` unchanged, and every config
    here sets ``lr_decay_iters`` equal to ``max_iters``, so the last iteration's
    rate is ``min_lr`` whichever side of the boundary it falls on.
    """
    cfg = config_values(config, repo)
    peak = as_number(cfg.get("learning_rate", ("", ""))[0])
    expr = cfg.get("min_lr", ("", ""))[0]
    floor = as_number(expr)
    if floor is None and peak is not None and "learning_rate" in expr:
        floor = as_number(expr.replace("learning_rate", repr(peak)))
    decay = cfg.get("lr_decay_iters", ("", ""))[0]
    iters = nanogpt_last_iter(run) or cfg.get("max_iters", ("", ""))[0]
    return {
        "modality": modality, "arch": "gpt2", "run": run.name,
        "steps_run": iters,
        "schedule": f"cosine to min_lr over lr_decay_iters={decay}, then flat",
        "peak_lr": f"{peak:.3e}" if peak is not None else "",
        "schedule_end_lr": f"{floor:.3e}" if floor is not None else expr,
        "ends_at_zero": "yes" if floor == 0 else "no",
        "last_logged_lr": "（nanoGPT は TensorBoard にしか書かない）",
        "last_logged_step": "",
        "config": config.name,
        "config_lines": "; ".join(v[1] for k, v in sorted(cfg.items())
                                  if k in ("min_lr", "learning_rate",
                                           "lr_decay_iters", "max_iters")),
        "source": "molcrawl/models/gpt2/train.py:845 get_lr + the config lines",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True,
                    help="the checkout the config paths are read from")
    ap.add_argument("--bert", action="append", default=[], nargs=3,
                    metavar=("MODALITY", "RUN_DIR", "CONFIG"))
    ap.add_argument("--gpt2", action="append", default=[], nargs=3,
                    metavar=("MODALITY", "RUN_DIR", "CONFIG"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    for modality, run, config in args.bert:
        rows.append(bert_row(modality, Path(run), Path(config), args.repo))
    for modality, run, config in args.gpt2:
        rows.append(gpt2_row(modality, Path(run), Path(config), args.repo))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # BOM so the file is not taken for binary when it is shared on Slack.
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    for row in rows:
        print(f"  {row['modality']:<18}{row['arch']:<6}"
              f"end={row.get('schedule_end_lr', ''):<12}"
              f"zero={row.get('ends_at_zero', ''):<4}{row.get('schedule', '')}")
    print(f"\n{len(rows)} 行 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
