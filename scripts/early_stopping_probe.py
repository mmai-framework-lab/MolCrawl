"""Probe whether early stopping stops GPT-2 and BERT where its rule says it should.

Every production run so far has had early stopping off, so the code path has never
been exercised on this cluster. This probe runs it end to end, on four GPUs, through
the real entry points (``models/gpt2/train.py`` and ``models/bert/main.py``).

How the expected stop is made exact
-----------------------------------
The learning rate is 0, so the weights never move, and the evaluation is fixed
(``deterministic_val_eval`` on GPT-2, ``fixed_eval_mask`` on BERT), so every
evaluation returns the same loss. The first evaluation is then the only
improvement, and the run must stop on the ``patience``-th evaluation after the one
that set the best. That gives one step number per case, written down before the
run, in ``CASES`` below.

Two checks per phase, so a surprise is told apart from a misread rule:

* ``expected``: the step written down in advance from the rule as read in the code.
* ``replay``: the rule re-applied to the losses the run actually logged. If the
  losses moved (they should not), ``expected`` no longer applies, but ``replay``
  still checks that the stop matches the rule on the values the run saw.

Plus ``rc`` -- a rank left behind at an NCCL collective shows up as the per-case
timeout (rc 124), not as a wrong step.

Usage (from the job script)::

    python scripts/early_stopping_probe.py write-configs OUT   # configs + plan.tsv
    python scripts/early_stopping_probe.py check OUT           # results.tsv, rc 1 on FAIL
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

PATIENCE = 3
EVAL_EVERY = 10  # GPT-2 eval_interval and BERT eval_steps (= log_interval)


# --------------------------------------------------------------------------- rules


def gpt2_replay(evals, patience, best=1e9, counter=0):
    """Where ``train.py`` stops, given ``(iter, val)`` pairs in the order evaluated.

    ``train.py`` (L1225-1241): a strictly lower val loss resets the counter,
    anything else adds one, and the run breaks once the counter reaches
    ``patience``. ``best`` and ``counter`` are what a resumed run starts from.
    Returns ``(stop_iter or None, counter, best)`` -- the last two are what a
    checkpoint taken at the end would carry.
    """
    for it, val in evals:
        if val < best:
            best, counter = val, 0
        else:
            counter += 1
            if counter >= patience:
                return it, counter, best
    return None, counter, best


def hf_replay(evals, patience, save_steps, save_on_improve=False,
              best_metric=None, counter=0, soi_best=None):
    """Where HF's ``EarlyStoppingCallback`` stops (transformers 4.45.1).

    The callback compares against ``state.best_metric``, and in 4.45.1 that field
    is written only inside ``_save_checkpoint`` -- so a new best counts as one only
    once a checkpoint has been saved at it. Per evaluation, in the order the
    Trainer runs them:

    1. ``on_evaluate``: no ``best_metric`` yet, or strictly lower than it -> the
       counter resets; otherwise +1. The run is told to stop at ``patience``.
    2. A save happens when the step is a multiple of ``save_steps``, or, with
       ``save_on_improve``, when the value is strictly below the best that
       callback has seen. A save writes ``best_metric`` if the value beats it.

    Returns ``(stop_step or None, counter, best_metric, soi_best)``.
    """
    for step, val in evals:
        if best_metric is None or val < best_metric:
            counter = 0
        else:
            counter += 1
        save = step % save_steps == 0
        if save_on_improve and (soi_best is None or val < soi_best):
            soi_best = val
            save = True
        if save and (best_metric is None or val < best_metric):
            best_metric = val
        if counter >= patience:
            return step, counter, best_metric, soi_best
    return None, counter, best_metric, soi_best


# --------------------------------------------------------------------------- cases


@dataclass
class Phase:
    """One launch. A case with two phases resumes in the second from the first."""

    name: str
    overrides: dict


@dataclass
class Case:
    name: str
    arch: str  # "gpt2" | "bert"
    phases: list
    expected: int  # last evaluated step of the last phase
    why: str
    alternative: dict = field(default_factory=dict)  # {"label": step} for a reading we do not expect


CASES = [
    Case("g_on", "gpt2", [Phase("a", {"early_stopping": True, "max_iters": 100})],
         expected=30,
         why="eval at 0 sets the best (from 1e9); 10/20/30 equal it -> counter 3 at 30"),
    Case("g_off", "gpt2", [Phase("a", {"early_stopping": False, "max_iters": 100})],
         expected=100,
         why="disabled: runs to max_iters and evaluates there"),
    Case("g_resume", "gpt2",
         [Phase("a", {"early_stopping": True, "max_iters": 20}),
          Phase("b", {"early_stopping": True, "max_iters": 100, "init_from": "resume"})],
         expected=20,
         why="a ends at 20 with counter 2, saved in checkpoint-20; b reloads it and "
             "re-evaluates step 20 (the loop evaluates before it steps) -> counter 3 at 20. "
             "Step 20 is counted twice: once in a, once in b",
         alternative={"counter reset on resume": 40}),
    Case("b_on", "bert", [Phase("a", {"early_stopping": True, "max_steps": 100})],
         expected=40,
         why="eval at 10 has no best_metric yet (counter 0), the save at 10 sets it; "
             "20/30/40 equal it -> counter 3 at 40"),
    Case("b_off", "bert", [Phase("a", {"early_stopping": False, "max_steps": 100})],
         expected=100,
         why="disabled: runs to max_steps"),
    Case("b_save20", "bert",
         [Phase("a", {"early_stopping": True, "max_steps": 100, "save_steps": 20})],
         expected=50,
         why="best_metric is written only at a save: 10 and 20 both see None (counter 0), "
             "the save at 20 sets it; 30/40/50 -> counter 3 at 50, one eval later than b_on",
         alternative={"best tracked at every eval": 40}),
    Case("b_save20_soi", "bert",
         [Phase("a", {"early_stopping": True, "max_steps": 100, "save_steps": 20,
                      "save_on_improve": True})],
         expected=40,
         why="save_on_improve saves at 10 (first value), which sets best_metric at 10 -> "
             "same as b_on although save_steps is 20",
         alternative={"save_on_improve not setting best_metric": 50}),
    Case("b_resume", "bert",
         [Phase("a", {"early_stopping": True, "max_steps": 30}),
          Phase("b", {"early_stopping": True, "max_steps": 100})],
         expected=60,
         why="a ends at 30 with counter 2. main.py does not set "
             "restore_callback_states_from_checkpoint, whose 4.45.1 default is False, so "
             "b restarts the counter at 0 while best_metric comes back from trainer_state: "
             "40/50/60 -> counter 3 at 60",
         alternative={"counter carried over": 40}),
]

GPT2_BASE = "molcrawl/tasks/pretrain/configs/compounds/gpt2_small.py"
BERT_BASE = "molcrawl/tasks/pretrain/configs/compounds/bert_small.py"

# Shared by every case. lr 0 keeps the weights fixed; the fixed evaluation keeps
# the loss fixed; the batch is cut down because the batch does not enter the rule.
GPT2_COMMON = {
    "learning_rate": 0.0, "min_lr": 0.0, "warmup_iters": 0, "lr_decay_iters": 100,
    "eval_interval": EVAL_EVERY, "log_interval": EVAL_EVERY, "eval_sequences": 320,
    "deterministic_val_eval": True, "early_stopping_patience": PATIENCE,
    "gradient_accumulation_steps": 4, "expected_global_batch": 64,  # 16 x 4
    "always_save_checkpoint": False, "max_checkpoints": 3,
    "compile": False, "tensorboard": False, "init_from": "scratch",
}
BERT_COMMON = {
    "learning_rate": 0.0, "warmup_steps": 0, "lr_scheduler_type": "constant",
    "log_interval": EVAL_EVERY, "save_steps": EVAL_EVERY, "save_on_improve": False,
    "fixed_eval_mask": True, "eval_mask_seed": 42,
    "early_stopping_patience": PATIENCE,
    "gradient_accumulation_steps": 1, "expected_global_batch": 128,  # 32 x 1 x 4
    "checkpoint_keep_best": 2, "checkpoint_keep_latest": 2,
}


def write_configs(out):
    """Write one config per phase and plan.tsv for the job script to walk.

    Configs rather than --key=value: BERT's configurator only accepts names that
    already exist as globals, and early_stopping_patience, fixed_eval_mask and the
    retention counts are read with globals().get(), so they can only come from a
    config file.
    """
    cfg_dir = os.path.join(out, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    rows = []
    for case in CASES:
        run_dir = os.path.join(out, "runs", case.name)
        base, common = (GPT2_BASE, GPT2_COMMON) if case.arch == "gpt2" else (BERT_BASE, BERT_COMMON)
        module = base[:-3].replace("/", ".")
        for ph in case.phases:
            values = dict(common, **ph.overrides)
            if case.arch == "gpt2":
                values["out_dir"] = run_dir
                values["tensorboard_dir"] = run_dir
            else:
                values["model_path"] = run_dir
            lines = [f"# early-stopping probe: case {case.name}, phase {ph.name}",
                     f"# expected last evaluated step: {case.expected} -- {case.why}",
                     f"from {module} import *  # noqa: F401,F403", ""]
            lines += [f"{k} = {v!r}" for k, v in values.items()]
            lines += ["seed = 42", ""]
            path = os.path.join(cfg_dir, f"{case.name}-{ph.name}.py")
            with open(path, "w") as f:
                f.write("\n".join(lines))
            rows.append((case.name, ph.name, case.arch, path, run_dir))
    with open(os.path.join(out, "plan.tsv"), "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"wrote {len(rows)} configs and {out}/plan.tsv")


# --------------------------------------------------------------------------- reading


def read_gpt2_evals(run_dir):
    """``(iter, val)`` per phase, from the per-launch logging_*.csv files.

    train.py opens a new timestamped CSV per launch (every rank creates the same
    name; only rank 0 writes rows), so the files sorted by name are the phases in
    order. Empty files from launches that died before evaluating are dropped.
    """
    phases = []
    for path in sorted(glob.glob(os.path.join(run_dir, "logging_*.csv"))):
        rows = []
        with open(path) as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3 and parts[0].isdigit():
                    rows.append((int(parts[0]), float(parts[2])))
        if rows:
            phases.append(rows)
    return phases


# One dict per evaluation, printed by HF's ProgressCallback on rank 0. Training-loss
# dicts carry 'loss' and never 'eval_loss_mask'.
_EVAL_LINE = re.compile(r"^\{.*'eval_loss_mask'.*\}$")
_RESUME_LINE = re.compile(r"Resuming training from: .*checkpoint-(\d+)")


def read_bert_evals(log_text, eval_steps=EVAL_EVERY):
    """``(step, eval_loss_mask)`` from one launch's stdout, plus the step it resumed at.

    HF prints each evaluation's metrics as a dict on the main process only, without
    the step, so the step is the resume point plus ``eval_steps`` per printed dict.
    Cross-checked against trainer_state.json in ``check`` where a checkpoint exists.
    """
    start = 0
    m = _RESUME_LINE.search(log_text)
    if m:
        start = int(m.group(1))
    evals = []
    for line in log_text.splitlines():
        if _EVAL_LINE.match(line.strip()):
            d = ast.literal_eval(line.strip())
            if "eval_loss_mask" in d:
                evals.append((start + eval_steps * (len(evals) + 1), float(d["eval_loss_mask"])))
    return start, evals


def latest_trainer_state(run_dir):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    for c in reversed(ckpts):
        p = os.path.join(c, "trainer_state.json")
        if os.path.exists(p):
            with open(p) as f:
                return int(c.rsplit("-", 1)[1]), json.load(f)
    return None, None


def _find_key(node, key):
    if isinstance(node, dict):
        if key in node and not isinstance(node[key], dict):
            return node[key]
        for v in node.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def gpt2_manifest_flag(run_dir):
    """``early_stopping`` as the last launch recorded it in run_manifest.json."""
    p = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return _find_key(json.load(f), "early_stopping")


# --------------------------------------------------------------------------- check


def _read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def check(out):
    results = []

    def add(case, item, ok, detail):
        verdict = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
        results.append((case, item, verdict, detail))

    for case in CASES:
        run_dir = os.path.join(out, "runs", case.name)
        logs = {ph.name: _read(os.path.join(out, "logs", f"{case.name}-{ph.name}.log")) for ph in case.phases}
        for ph in case.phases:
            rc = _read(os.path.join(out, "rc", f"{case.name}-{ph.name}")).strip()
            add(case.name, f"rc[{ph.name}]", rc == "0",
                f"rc={rc or 'missing'}" + (" (timeout: a rank was left waiting?)" if rc == "124" else ""))

        if case.arch == "gpt2":
            phases = read_gpt2_evals(run_dir)
            if len(phases) != len(case.phases):
                add(case.name, "evals", False, f"{len(phases)} logging CSVs with rows, expected {len(case.phases)}")
                continue
            # train.py stores best_val_loss and the counter in the checkpoint and
            # reads both back on resume, so the replay carries them across phases.
            best, counter, stop = 1e9, 0, None
            for ev in phases:
                stop, counter, best = gpt2_replay(ev, PATIENCE, best=best, counter=counter)
            last_max = case.phases[-1].overrides["max_iters"]
            actual = phases[-1][-1][0]
            values = [v for ev in phases for _, v in ev]
            triggered = "Early stopping triggered" in logs[case.phases[-1].name]
            flag = gpt2_manifest_flag(run_dir)
            want_flag = case.phases[-1].overrides["early_stopping"]
            add(case.name, "manifest early_stopping", flag is want_flag, f"{flag}")
            add(case.name, "stop message", triggered == (case.expected < 100),
                "printed" if triggered else "not printed")
        else:
            phases, starts = [], []
            for ph in case.phases:
                start, ev = read_bert_evals(logs[ph.name])
                starts.append(start)
                phases.append(ev)
            if not all(phases):
                add(case.name, "evals", False, f"evaluations per phase: {[len(p) for p in phases]}")
                continue
            save_steps = case.phases[-1].overrides.get("save_steps", BERT_COMMON["save_steps"])
            soi = case.phases[-1].overrides.get("save_on_improve", BERT_COMMON["save_on_improve"])
            # A resumed launch gets best_metric back from trainer_state but starts the
            # counter at 0 (restore_callback_states_from_checkpoint=False), and
            # SaveOnMetricImprovement starts with no best of its own.
            best_metric, stop = None, None
            for ev in phases:
                stop, _, best_metric, _ = hf_replay(ev, PATIENCE, save_steps, soi, best_metric=best_metric)
            last_max = case.phases[-1].overrides["max_steps"]
            actual = phases[-1][-1][0]
            values = [v for ev in phases for _, v in ev]
            if len(case.phases) > 1:
                add(case.name, "resumed at", starts[-1] == phases[0][-1][0],
                    f"phase b resumed from checkpoint-{starts[-1]}")
            ck_step, state = latest_trainer_state(run_dir)
            if state is not None:
                # log_history survives a resume, so it holds every phase up to ck_step.
                logged = [e["step"] for e in state.get("log_history", []) if "eval_loss_mask" in e]
                parsed = [s for ev in phases for s, _ in ev if s <= ck_step]
                add(case.name, "steps vs trainer_state", logged == parsed,
                    f"checkpoint-{ck_step} log_history {logged}, read from stdout {parsed}")
                es = (state.get("stateful_callbacks") or {}).get("EarlyStoppingCallback")
                want = case.phases[-1].overrides["early_stopping"]
                if want:
                    detail = json.dumps(es) if es else "absent"
                    add(case.name, "callback in trainer_state", es is not None, f"checkpoint-{ck_step}: {detail}")
                else:
                    add(case.name, "callback in trainer_state", es is None, "absent" if es is None else json.dumps(es))

        constant = len(set(values)) == 1
        add(case.name, "loss constant (lr 0)", True if constant else None,
            f"{len(values)} evals, {len(set(values))} distinct value(s), first {values[0]:.6f}")
        exp_ok = actual == case.expected
        alt = [k for k, s in case.alternative.items() if s == actual]
        add(case.name, "last eval vs expected", exp_ok if constant else None,
            f"actual {actual}, expected {case.expected}" + (f" -- matches '{alt[0]}'" if alt else ""))
        want = stop if stop is not None else last_max
        add(case.name, "last eval vs rule replay", want == actual,
            f"rule on the logged losses ends at {want}"
            f"{' (no stop: max)' if stop is None else ''}, run ended at {actual}")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["case", "check", "verdict", "detail"])
        w.writerows(results)
    width = max(len(r[0]) for r in results)
    for r in results:
        print(f"{r[0]:<{width}}  {r[2]:<4}  {r[1]}: {r[3]}")
    fails = sum(r[2] == "FAIL" for r in results)
    print(f"\n{len(results)} checks, {fails} FAIL -> {out}/results.tsv")
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("write-configs", "check"):
        sub.add_parser(name).add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "write-configs":
        write_configs(a.out)
        return 0
    return check(a.out)


if __name__ == "__main__":
    sys.exit(main())
