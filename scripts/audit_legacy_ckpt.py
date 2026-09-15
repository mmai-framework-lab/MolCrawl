#!/usr/bin/env python3
"""Say whether each ``ckpt.pt`` holds the best weights or merely the last ones.

Directive 2026-09-15 §6.3. ``models/gpt2/train.py`` writes ``ckpt.pt`` under

    if keep_legacy_ckpt or is_best_model:

so with the flag off the file is the best-val state, and with it on the periodic
saves overwrite it and the file is whatever was written last. A score read off a
``ckpt.pt`` from a run that had the flag on is not the score of the best
checkpoint, and `workflows/eval-protein-old-vs-new-uncapped.sh` records that the
2026-03-16 protein run was trained with it on.

The flag cannot be recovered from the run's own record: it is not in the tracked
list in ``models/gpt2/_run_manifest.py``, and it was frequently passed at launch
rather than set in a config. What can be recovered is the consequence, which is
what actually matters -- compare the iteration inside ``ckpt.pt`` against the
step the minimum validation loss was recorded at in the same directory's
``logging_*.csv``. They agree exactly when the file is the best-val state.

Reading is done without materialising tensors. A torch save file is a zip whose
``data.pkl`` holds the structure and whose storages are separate members, so
unpickling with a ``persistent_load`` that returns placeholders gives the scalar
fields -- ``iter_num``, ``best_val_loss`` -- at the cost of reading a few KB
instead of the whole model and optimizer.
"""

import argparse
import csv
import io
import os
import pickle
import sys
import zipfile


class _Stub:
    """Stands in for any class the audit does not need to reconstruct."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return None

    def append(self, item):          # pickle APPEND/APPENDS on a stubbed list
        pass

    def __setitem__(self, key, value):
        pass


class _NoTensors(pickle.Unpickler):
    """Unpickle the structure, standing a placeholder in for everything else.

    The audit reads three scalars out of the top-level dict. Everything around
    them -- torch storages, the rng_state's numpy arrays, whatever a config
    happened to hold -- is stubbed rather than reconstructed, so the audit runs
    wherever python does and does not depend on torch or numpy being importable.
    Rebuilding them would also mean reading the tensor data this exists to avoid.
    """

    def persistent_load(self, pid):
        return None

    def find_class(self, module, name):
        if module.startswith(("torch", "numpy")):
            return _Stub
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub


def read_scalars(path):
    """``{iter_num, best_val_loss, best_val_step}`` from a torch save file."""
    with zipfile.ZipFile(path) as zf:
        name = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
        if name is None:
            raise ValueError("no data.pkl; not a torch zip checkpoint")
        obj = _NoTensors(io.BytesIO(zf.read(name))).load()
    if not isinstance(obj, dict):
        raise ValueError(f"top level is {type(obj).__name__}, expected dict")
    return {k: obj.get(k) for k in ("iter_num", "best_val_loss", "best_val_step")}


def min_eval(directory):
    """``(steps, val_loss)`` of the lowest validation loss this run recorded.

    ``steps`` is every step that reached that value, earliest first, because the
    log rounds to four decimals and the run did not. compounds gpt2-small printed
    0.5944 at both 1500 and 1550, and the run -- comparing in full precision --
    took 1550. Treating the first as "the" minimum makes the checkpoint look stale
    when it is exactly right, so a tie is reported as a tie.
    """
    best_val = None
    steps = []
    for entry in sorted(os.listdir(directory)):
        if not (entry.startswith("logging_") and entry.endswith(".csv")):
            continue
        with open(os.path.join(directory, entry), newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    step, val = int(row[0]), float(row[2])
                except ValueError:
                    continue
                if best_val is None or val < best_val:
                    best_val, steps = val, [step]
                elif val == best_val:
                    steps.append(step)
    return sorted(steps), best_val


def _as_loss(value):
    """Format a loss field, saying so when it is not a plain number.

    train.py assigns ``best_val_loss = losses["val"]``, a 0-dim tensor, and its
    value lives in a storage this audit deliberately does not read. The column is
    context; the verdict rests on iter_num against the eval log, and the log is
    where the authoritative number is anyway.
    """
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "tensor (not read)"


def find_checkpoints(roots):
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if "ckpt.pt" in filenames:
                yield root, dirpath


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=None,
                    help="directory trees to walk; defaults to $AUDIT_ROOTS (colon-separated)")
    args = ap.parse_args(argv)

    roots = args.roots or [r for r in os.environ.get("AUDIT_ROOTS", "").split(":") if r]
    if not roots:
        print("no roots given: pass them as arguments or set AUDIT_ROOTS "
              "(colon-separated) to the learning_source trees to walk", file=sys.stderr)
        return 2
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        print(f"not a directory: {', '.join(missing)}", file=sys.stderr)
        return 2

    print("| run | ckpt.pt iter | min-val step | verdict | ckpt best_val_loss | min val |")
    print("|---|---|---|---|---|---|")
    seen = 0
    for root, directory in find_checkpoints(roots):
        seen += 1
        rel = os.path.relpath(directory, root)
        path = os.path.join(directory, "ckpt.pt")
        try:
            fields = read_scalars(path)
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            print(f"| {rel} | - | - | unreadable: {exc.__class__.__name__}: {exc} | - | - |")
            continue
        steps, val = min_eval(directory)
        it = fields.get("iter_num")
        if not steps:
            verdict, shown = "no eval log — undecidable", "-"
        elif it is None:
            verdict, shown = "no iter_num in ckpt.pt — undecidable", str(steps[0])
        elif it in steps:
            shown = "/".join(str(s) for s in steps)
            verdict = "best" if len(steps) == 1 else "best (tied at log precision)"
        else:
            shown = "/".join(str(s) for s in steps)
            verdict = f"**LAST, not best** (off by {it - steps[-1]:+d})"
        print(f"| {rel} | {it} | {shown} | {verdict} | "
              f"{_as_loss(fields.get('best_val_loss'))} | "
              f"{'-' if val is None else f'{val:.4f}'} |")
    if seen == 0:
        print(f"\nNo ckpt.pt found under: {', '.join(roots)}")
    else:
        print(f"\n{seen} checkpoint(s) examined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
