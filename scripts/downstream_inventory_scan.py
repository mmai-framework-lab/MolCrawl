"""List the downstream evaluations, each tied to the pretraining run it scored.

One row per evaluation x model. The point of the row is the join: ``tree`` and
``run`` name the pretraining run the way ``run_inventory_scan.py`` names it, so
the two tables meet on that pair and a reader can go from a downstream score to
the run that produced the checkpoint.

Only sources that *record* the run are read. Three do:

  * the compounds TDC results write ``models.<name>.run_dir`` and
    ``adopted_checkpoint`` into each task's JSON, with the untrained seeds and
    the fingerprint baseline beside them in the same file,
  * the molecule/natural-language retrieval results write ``model_path``, down
    to the checkpoint,
  * the ProteinGym tables are keyed by (arch, size), and the run directory for
    that pair lives in ``protein_proteingym_tables.MODELS`` -- imported rather
    than copied, so the two cannot drift apart.

Three others identify the model by (arch, size, lr) and a step, which does not
single out a run: protein alone has three runs at bert/small/4.1e-4 (the base
and seeds 1 and 17) and two at bert/large/3.5e-5 (the base and a rerun). Those
evaluations are left out until their generators record the run, rather than
guessed at from the numbers.

    python scripts/downstream_inventory_scan.py --config <file.json> --out <file.tsv>

Paths live in the config file, not here: this repository is public.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

CKPT_RE = re.compile(r"^checkpoint-\d+$")
UNTRAINED_RE = re.compile(r"-untrained-seed\d+$")

COLUMNS = ["modality", "eval", "metric", "better", "arch", "size", "tree", "run",
           "checkpoint", "value", "sd", "control", "control_value",
           "baseline", "baseline_value", "n", "set", "source"]

# The compounds JSONs use the task's own slug; these are what a report calls them.
TASK_LABEL = {"herg": "hERG 阻害", "solubility": "水溶解度", "smoke": "疎通確認"}
DIRECTION_LABEL = {"molecule_to_text": "検索 分子→説明文",
                   "text_to_molecule": "検索 説明文→分子"}


def split_run(path: str, strip: str, prefix: str = "learning_source"):
    """(tree, run, checkpoint) from an absolute model path.

    The path may name the run directory, a ``checkpoint-N`` inside it, or
    nanoGPT's ``ckpt.pt`` -- the retrieval runs use all three.

    The tree is found by name rather than by cutting a fixed prefix, because not
    every path comes from the same checkout: the result JSONs were written by
    the main one, while the ProteinGym run directories are computed from
    ``protein_proteingym_tables.__file__`` and so carry whichever worktree ran
    this scan. Cutting ``strip`` left those paths whole and the join lost all
    eight protein rows.
    """
    p = Path(path)
    checkpoint = ""
    if p.name == "ckpt.pt" or CKPT_RE.match(p.name):
        checkpoint, p = p.name, p.parent
    parts = p.parts
    for i, part in enumerate(parts):
        if part.startswith(prefix):
            return part, str(Path(*parts[i + 1:])) if i + 1 < len(parts) else "", checkpoint
    rel = str(p)
    if strip and rel.startswith(strip):
        rel = rel[len(strip):]
    parts = Path(rel).parts
    if not parts:
        return "", rel, checkpoint
    return parts[0], str(Path(*parts[1:])) if len(parts) > 1 else "", checkpoint


def row(**kw):
    r = {c: "" for c in COLUMNS}
    r.update(kw)
    return r


def fmt(v, nd=4):
    return "" if v is None else f"{float(v):.{nd}f}"


def read_compounds(spec, report: Path, strip: str, rows):
    """A task's JSON carries the model, the untrained seeds and the fingerprint."""
    trained = report / spec["dir"]
    control = report / spec["control_dir"] if spec.get("control_dir") else None
    for jf in sorted(trained.glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        models = d.get("models", {})
        ctl = {}
        if control and (control / jf.name).exists():
            ctl = json.loads((control / jf.name).read_text(encoding="utf-8")).get("models", {})
        fingerprint = next((v for k, v in {**models, **ctl}.items()
                            if v.get("arch") == "none"), {})
        fp_name = next((k for k, v in {**models, **ctl}.items()
                        if v.get("arch") == "none"), "")
        for name, m in models.items():
            if m.get("arch") == "none" or UNTRAINED_RE.search(name) or not m.get("run_dir"):
                continue
            # The untrained arm is three initialisations of the same shape; the
            # control is their mean, which is what the improvement is read against.
            seeds = [v["mean"] for k, v in ctl.items()
                     if UNTRAINED_RE.search(k) and k.startswith(name) and "mean" in v]
            tree, run, ck = split_run(m["run_dir"], strip)
            rows.append(row(
                modality=spec["modality"],
                eval=TASK_LABEL.get(d.get("task", ""), d.get("task", "")),
                metric=d.get("metric", ""), better="higher",
                arch=m.get("arch", ""), size=m.get("size", ""),
                tree=tree, run=run, checkpoint=m.get("adopted_checkpoint", "") or ck,
                value=fmt(m.get("mean")), sd=fmt(m.get("sd")),
                control="未学習 3 seed 平均" if seeds else "",
                control_value=fmt(sum(seeds) / len(seeds)) if seeds else "",
                baseline=fp_name, baseline_value=fmt(fingerprint.get("mean")),
                n=str(m.get("scored", "") or d.get("molecules", "") or ""),
                set=spec.get("set", ""), source=f"{spec['dir']}/{jf.name}"))


def read_retrieval(spec, report: Path, strip: str, rows):
    """One row per direction: the two directions are separate evaluations."""
    for jf in sorted((report / spec["dir"]).glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        if not d.get("model_path"):
            continue
        tree, run, ck = split_run(d["model_path"], strip)
        # A format variant lives in the file name after "__"; it is part of the
        # evaluation, not of the model.
        variant = jf.stem.split("__", 1)[1] if "__" in jf.stem else ""
        for direction, m in (d.get("directions") or {}).items():
            label = DIRECTION_LABEL.get(direction, direction)
            rows.append(row(
                modality=spec["modality"],
                eval=f"{label}（{variant}）" if variant else label,
                metric="recall@1", better="higher",
                arch=d.get("arch", ""), size=d.get("size", ""),
                tree=tree, run=run, checkpoint=ck,
                value=fmt(m.get("recall@1")),
                control="偶然", control_value=fmt(m.get("random_baseline_recall@1")),
                baseline="MRR", baseline_value=fmt(m.get("mrr")),
                n=str(m.get("queries", "") or ""),
                set=spec.get("set", ""), source=f"{spec['dir']}/{jf.name}"))


def read_proteingym(spec, report: Path, strip: str, rows):
    """(arch, size) -> run directory comes from the table script's own MODELS."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protein_proteingym_tables import MODELS  # noqa: E402  (one place names the runs)
    predir = {(a, s): (d, k) for a, s, _lr, _seed, _slug, d, _lm, k in MODELS}
    with open(report / spec["per_model"], encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            got = predir.get((r["arch"], r["size"]))
            if got is None:
                continue
            d, kind = got
            tree, run, _ck = split_run(str(d), strip)
            # The table's "checkpoint" is a step. Only the HF runs keep a
            # directory of that name; nanoGPT overwrites one ckpt.pt, so naming a
            # checkpoint-N there would point at something that does not exist.
            step = r.get("checkpoint", "")
            ck = (f"checkpoint-{step}" if kind == "bert" else f"step {step}") if step else ""
            done = r.get("status") == "done"
            rows.append(row(
                modality=spec["modality"], eval="ProteinGym", metric="Spearman",
                better="higher", arch=r["arch"], size=r["size"],
                tree=tree, run=run, checkpoint=ck,
                value=r["spearman_mean"] if done else "",
                sd="", control="", control_value="",
                baseline="中央値", baseline_value=r["spearman_median"] if done else "",
                n=r.get("n_assay_done", ""),
                set=spec.get("set", ""),
                source=spec["per_model"] if done else f"{spec['per_model']}（{r['status']}）"))


READERS = {"compounds_tdc": read_compounds,
           "molnl_retrieval": read_retrieval,
           "proteingym": read_proteingym}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = Path(cfg["report_dir"])
    strip = cfg.get("strip", "")

    rows: list[dict] = []
    for spec in cfg["sources"]:
        reader = READERS.get(spec["kind"])
        if reader is None:
            print(f"  unknown kind {spec['kind']!r}, skipped", file=sys.stderr)
            continue
        before = len(rows)
        reader(spec, report, strip, rows)
        print(f"  {spec['kind']:16s} {spec.get('dir') or spec.get('per_model')}"
              f"  {len(rows) - before} 行")

    rows.sort(key=lambda r: (r["modality"], r["eval"], r["arch"], r["run"]))
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    runs = {(r["tree"], r["run"]) for r in rows}
    print(f"wrote {args.out} ({len(rows)} 行 / {len(runs)} run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
