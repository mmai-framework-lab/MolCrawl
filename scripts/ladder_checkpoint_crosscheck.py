#!/usr/bin/env python3
"""Put the paper's size ladder beside the checkpoint-selection error, rung by rung.

The 2026-11-02b ladder-check directive asks one question: where a ladder rung's adopted
checkpoint is not the step at which the judged metric bottomed out, is the gap to the
neighbouring size of the same order as that error? So every rung needs four numbers --
adopted step and value, the judged metric's minimum step and value -- and the difference
to the size below it.

Nothing here decides which run a rung is. That is a research choice and it lives in
``scripts/ladder_runs.tsv`` with its source. This script reads the runs:

  - BERT: ``best_model_checkpoint`` and ``best_metric`` against the minimum over the whole
    ``log_history``, as ``audit_checkpoint_metric_match.py`` already computes them. A rung
    is in the 42 when that audit calls the run production and the two disagree.
  - GPT-2: nanoGPT keeps its own ``best_val_loss`` and writes a checkpoint when it
    improves, so the adopted step is read from the ``checkpoint-*/training_args.json`` with
    the lowest ``val_loss`` and compared with the minimum of the logged curve. The 42 are
    BERT runs; this path is reported so the comparison is not silently one-architecture.

The logged GPT-2 curve carries four decimals, so the two are compared by step, not value:
a value tie across several iterations is rounding, not a real tie.

    python scripts/ladder_checkpoint_crosscheck.py \
      --audit-json report/checkpoint-metric-match-20261102.json \
      --ladder scripts/ladder_runs.tsv \
      --mismatches report/checkpoint-metric-mismatches-20261102.tsv \
      --out report/ladder-checkpoint-crosscheck-2026-11-02.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

SIZE_ORDER = ("small", "medium", "large", "xl", "ex-large")

# Half of the last place the nanoGPT logging CSV keeps.
CSV_ROUNDING = 5e-5

# The directive's own §1 breakdown of the 42, by the names it uses. The 27 that learn are
# given there as counts only, so they are what is left once these 15 are taken out; the
# counts are checked against the directive's rna 9 / molnl 8 / protein 6 / compounds 4.
DIRECTIVE_CLASSES = {
    "学習していない": [
        ("protein_sequence", "protein_sequence-xl-lr5p5e6"),
        ("protein_sequence", "protein_sequence-xl-lr8e6"),
        ("protein_sequence", "protein_sequence-xl-lr1p2e5"),
        ("protein_sequence", "protein_sequence-xl-lr1p7e5"),
        ("protein_sequence", "protein_sequence-xl-lr2p4e5"),
        ("protein_sequence", "protein_sequence-xl-lr3p4e5"),
        ("protein_sequence", "protein_sequence-large-lr3e4"),
        ("protein_sequence", "protein_sequence-medium-lr1e3"),
        ("rna", "rna-large-lr1e3"),
        ("rna", "rna-xl-lr1e4"),
    ],
    "途中で崩れた": [
        ("compounds", "compounds-xl-lr1p5e4"),
        ("molecule_nat_lang", "bert_medium_24k_lr5p5e4"),
        ("protein_sequence", "protein_sequence-large-lr5e5"),
        ("protein_sequence", "protein_sequence-small-lr5p5e4"),
        ("rna", "rna-small-lr1e3"),
    ],
}
DIRECTIVE_LEARNING_COUNTS = {"rna": 9, "molecule_nat_lang": 8,
                             "protein_sequence": 6, "compounds": 4}

COLUMNS = [
    # The seven columns §7 names, in its order, then what §4 needs in order to put the
    # neighbouring rung's error beside the gap without opening a second file.
    "modality", "arch", "size", "run", "adopted_step", "adopted_value",
    "min_step", "min_value", "diff", "in_42", "neighbor_diff",
    "neighbor_size", "neighbor_run", "neighbor_diff_error", "own_diff_error",
    "judged_metric", "eval_interval", "save_interval", "off_grid_checkpoints",
    "steps_run", "seed",
    "on_ladder", "source", "note",
]


def read_tsv(path):
    """Rows of a TSV, skipping comment lines and tolerating a BOM."""
    with open(path, encoding="utf-8-sig") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def load_runs(path):
    """{(modality, run): row} from the audit JSON, production rows winning a clash."""
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)["runs"]
    index = {}
    for row in rows:
        key = (row["modality"], row["run"])
        if key in index and not row.get("production"):
            continue
        if key in index and index[key].get("production"):
            # Two trees hold a run of this name and both are production: keep the one
            # that ran further rather than whichever the walk reached first.
            if (row.get("steps_run") or 0) <= (index[key].get("steps_run") or 0):
                continue
        index[key] = row
    return index


def gpt2_adopted(run_dir):
    """The saved checkpoint with the lowest val loss: (step, value, checkpoints seen).

    Each nanoGPT checkpoint directory carries a small training_args.json with its own
    iteration and val_loss. No weight file is opened.
    """
    best = None
    seen = 0
    try:
        names = sorted(os.listdir(run_dir))
    except OSError:
        return None, None, 0
    for name in names:
        if not name.startswith("checkpoint-"):
            continue
        meta_path = os.path.join(run_dir, name, "training_args.json")
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            step = int(meta["iteration"])
            value = float(meta["val_loss"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        seen += 1
        if best is None or (value, step) < (best[1], best[0]):
            best = (step, value)
    if best is None:
        return None, None, seen
    return best[0], best[1], seen


def manifest_intervals(run_dir):
    """(eval_interval, save_interval) as the run itself recorded them.

    nanoGPT has no save interval: with always_save_checkpoint off it writes a checkpoint
    when the val loss improves and keeps the most recent max_checkpoints of them, which
    is why its adopted step is the minimum by construction.
    """
    try:
        with open(os.path.join(run_dir, "run_manifest.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return "", ""
    selection = manifest.get("selection") or {}
    evaluation = manifest.get("eval") or {}
    eval_interval = (selection.get("eval_steps")
                     or evaluation.get("eval_interval") or "")
    save_interval = (selection.get("save_steps")
                     or evaluation.get("save_checkpoint_steps") or "")
    if not save_interval and "always_save_checkpoint" in selection:
        keep = selection.get("max_checkpoints")
        save_interval = ("評価ごと" if selection["always_save_checkpoint"]
                         else "改善時のみ")
        if keep:
            save_interval += f"（保持 {keep}）"
    return eval_interval, save_interval


def off_grid_checkpoints(run_row, save_interval):
    """How many of the checkpoints on disk are not on the save grid.

    An off-grid checkpoint is save-on-improve at work: it is the only way a step that is
    not a multiple of save_steps gets written. The manifest does not record whether the
    callback was on, so this is read from what the run left behind.
    """
    try:
        interval = int(save_interval)
    except (TypeError, ValueError):
        return ""
    steps = run_row.get("checkpoints_present") or []
    if not steps:
        return ""
    return sum(1 for step in steps if int(step) % interval)


def number(value, places=6):
    return "" if value is None else f"{float(value):.{places}f}"


def rung(spec, run_row):
    """One output row: what the run says, before any comparison between sizes."""
    out = {key: "" for key in COLUMNS}
    out.update({
        "modality": spec["modality"], "arch": spec["arch"], "size": spec["size"],
        "run": spec["run"], "on_ladder": spec["on_ladder"],
        "source": spec.get("source", ""), "note": spec.get("note", ""),
    })
    if run_row is None:
        out["note"] = (out["note"] + " / 走査に無い run" if out["note"]
                       else "走査に無い run")
        return out

    eval_interval, save_interval = manifest_intervals(run_row["run_dir"])
    out["eval_interval"] = eval_interval
    out["save_interval"] = save_interval
    out["off_grid_checkpoints"] = off_grid_checkpoints(run_row, save_interval)
    out["steps_run"] = run_row.get("steps_run") or ""
    out["seed"] = "" if run_row.get("seed") is None else run_row["seed"]
    out["judged_metric"] = run_row.get("judged_on") or ""

    if run_row["arch"] == "bert":
        judged_min = run_row.get("judged_min") or {}
        adopted_step = run_row.get("best_model_checkpoint_step")
        adopted_value = run_row.get("best_metric_recorded")
        out["adopted_step"] = "" if adopted_step is None else adopted_step
        out["adopted_value"] = number(adopted_value)
        out["min_step"] = judged_min.get("step", "")
        out["min_value"] = number(judged_min.get("value"))
        if adopted_value is not None and judged_min.get("value") is not None:
            out["diff"] = number(float(adopted_value) - float(judged_min["value"]))
        out["in_42"] = ("yes" if (run_row.get("production")
                                  and run_row.get("match") == "no") else "no")
        return out

    curve = run_row.get("val_curve") or {}
    adopted_step, adopted_value, _ = gpt2_adopted(run_row["run_dir"])
    out["adopted_step"] = "" if adopted_step is None else adopted_step
    out["adopted_value"] = number(adopted_value)
    out["min_step"] = curve.get("min_iter", "")
    out["min_value"] = number(curve.get("min_value"), 4)
    if adopted_step is not None and curve.get("min_iter") is not None:
        if int(adopted_step) == int(curve["min_iter"]):
            out["diff"] = "0.000000"
        elif adopted_value is not None and (
                float(adopted_value) <= float(curve["min_value"]) + CSV_ROUNDING):
            # The logged curve keeps four decimals, so a step whose true value is higher
            # can tie with the adopted one there. The checkpoint's own value settles it.
            out["diff"] = f"0.000000（曲線の 4 桁丸めでは step {curve['min_iter']}）"
        else:
            out["diff"] = "step が違う"
    out["in_42"] = "no（42 本は BERT。GPT-2 はこの判定を通らない）"
    return out


def add_neighbours(rows):
    """Fill the gap to the size below, within one modality and architecture."""
    by_track = {}
    for row in rows:
        if row["on_ladder"] != "yes":
            continue
        by_track.setdefault((row["modality"], row["arch"]), []).append(row)
    for track_rows in by_track.values():
        track_rows.sort(key=lambda r: SIZE_ORDER.index(r["size"])
                        if r["size"] in SIZE_ORDER else len(SIZE_ORDER))
        for previous, current in zip(track_rows, track_rows[1:]):
            current["neighbor_size"] = previous["size"]
            current["neighbor_run"] = previous["run"]
            current["neighbor_diff_error"] = previous["diff"]
            current["own_diff_error"] = current["diff"]
            try:
                current["neighbor_diff"] = number(
                    float(current["adopted_value"]) - float(previous["adopted_value"]))
            except ValueError:
                current["neighbor_diff"] = ""


def check_directive_classes(mismatch_path, out=sys.stdout):
    """§1: does the directive's breakdown land on the 42 rows, one for one?"""
    rows = read_tsv(mismatch_path)
    keys = {(row["modality"], row["run"]) for row in rows}
    print(f"\n=== §1 42 本の切り分け（{mismatch_path}: {len(rows)} 行）", file=out)
    named = set()
    problems = []
    for label, entries in DIRECTIVE_CLASSES.items():
        missing = [f"{m}/{r}" for m, r in entries if (m, r) not in keys]
        named.update(entries)
        print(f"{label}: 指示書 {len(entries)} 本 / 42 本の中に "
              f"{len(entries) - len(missing)} 本", file=out)
        if missing:
            problems.append(f"{label}: 42 本に無い名前 {missing}")
    learning = [key for key in keys if key not in named]
    counts = {}
    for modality, _ in learning:
        counts[modality] = counts.get(modality, 0) + 1
    print(f"学習が成り立っている（差し引き）: {len(learning)} 本 {counts}", file=out)
    if len(learning) != 27:
        problems.append(f"差し引きが 27 本にならない: {len(learning)}")
    for modality, expected in DIRECTIVE_LEARNING_COUNTS.items():
        if counts.get(modality, 0) != expected:
            problems.append(f"{modality} の本数が指示書と違う: "
                            f"{counts.get(modality, 0)} 対 {expected}")
    for problem in problems:
        print(f"⚠️  {problem}", file=out)
    if not problems:
        print("指示書の内訳は 42 本にそのまま載る（27 + 10 + 5）", file=out)
    return not problems


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", required=True,
                        help="output of audit_checkpoint_metric_match.py")
    parser.add_argument("--ladder", default="scripts/ladder_runs.tsv",
                        help="which run each rung is, and where that is written down")
    parser.add_argument("--mismatches",
                        help="the 42-row TSV, for the §1 check against the directive")
    parser.add_argument("--out", required=True, help="TSV to write")
    args = parser.parse_args(argv)

    runs = load_runs(args.audit_json)
    rows = []
    for spec in read_tsv(args.ladder):
        if not spec.get("modality"):
            continue
        rows.append(rung(spec, runs.get((spec["modality"], spec["run"]))))
    add_neighbours(rows)
    write_tsv(args.out, rows)

    print(f"wrote {args.out} ({len(rows)} rows)")
    on_ladder = [row for row in rows if row["on_ladder"] == "yes"]
    print(f"梯子に載っている rung {len(on_ladder)} 本 / 表 {len(rows)} 行")
    flagged = [row for row in on_ladder if row["in_42"] == "yes"]
    print(f"そのうち 42 本に入っている rung: {len(flagged)}")
    for row in flagged:
        print(f"  {row['modality']:18s} {row['arch']:5s} {row['size']:9s} "
              f"{row['run']:40s} 差 {row['diff']}")

    print("\n=== §4 隣との差と誤差")
    for row in on_ladder:
        if not row["neighbor_size"]:
            continue
        print(f"{row['modality']:18s} {row['arch']:5s} "
              f"{row['neighbor_size']:9s}→{row['size']:9s} "
              f"隣との差 {row['neighbor_diff']:>12s} "
              f"誤差 {row['neighbor_diff_error'] or '-':>12s} / "
              f"{row['own_diff_error'] or '-':>12s}")

    print("\n=== §5 評価と保存の間隔")
    seen = {}
    for row in rows:
        if row["on_ladder"] != "yes" or not row["eval_interval"]:
            continue
        seen.setdefault((row["modality"], row["arch"]), set()).add(
            (str(row["eval_interval"]), str(row["save_interval"]),
             str(row["off_grid_checkpoints"])))
    for (modality, arch), pairs in sorted(seen.items()):
        for eval_interval, save_interval, off_grid in sorted(pairs):
            print(f"{modality:18s} {arch:5s} eval {eval_interval:>6s} / "
                  f"save {save_interval:<16s} 保存グリッド外の checkpoint "
                  f"{off_grid or '-'}")

    ok = True
    if args.mismatches:
        ok = check_directive_classes(args.mismatches)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
