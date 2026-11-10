#!/usr/bin/env python3
"""Split the 14 test-split retrieval measurements by whether the molecule is in train.

The 2026-11-10 judgment asks, in 3.2.1, whether the pairs whose molecule occurs in the
pretraining split score differently from the pairs whose molecule does not. If the two
sides read alike, then having seen the molecule did not make the retrieval work; if they
differ, the difference is the thing to report. The evaluation pairs are not changed and
nothing is scored again -- the per-query ranks are already on disk, and this only cuts
them in two.

In 3.2.2 it asks which tasks the matched molecules occur in. A molecule that only ever
appears in a retrosynthesis or a name-conversion record was seen without its
description; one that appears in a molecule_captioning or molecule_generation record was
seen next to a description, even if the wording differs.

Both answers need the same thing: which of the 1,000 pair positions have a molecule that
occurs in train, and in what kind of record. That list is not on disk. The earlier run
(report/molnl-smoltest-overlap.json) wrote only the count, 102 distinct canonical
molecules, so this scans for the positions and the task types in one pass.

What it reads
-------------
``arrow_splits/train.arrow`` rather than the packed corpus, for two reasons: it is the
only one carrying ``task_type``, and its ``input_text`` is plain text, so only the
response side has to be decoded. It holds 3,267,176 records against the packed corpus's
3,267,172 -- the four the packing dropped are in this scan and were not in the earlier
one, which is stated in the output rather than left to be noticed.

How a molecule is matched
-------------------------
The same criterion as the earlier overlap run: a side with no space in it is taken as
molecule-shaped, canonicalised with RDKit, and compared against the canonical form of
the pair's molecule. Only 9% of the corpus's own SMILES are already canonical, so
matching as written undercounts by more than half (44 against 102). The 83 molecules
RDKit will not read cannot be matched either way and are reported as their own bucket,
not folded silently into "not in train".

Checks that run every time
--------------------------
* the per-frame metrics recomputed over all 1,000 queries must equal the ones the
  scoring runs wrote, or the split is being taken off a different measurement;
* where two rank files hold the same frame -- the ``rt1-`` copies do -- they must agree
  row for row before either is used.

    python scripts/molnl_smoltest_train_overlap_split.py --arrow-train <dir> \
        --pairs-file <jsonl> --ranks-dir <dir> --out <json> --tsv <tsv>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from array import array
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool

MOLECULE_TASKS = ("molecule_captioning", "molecule_generation")
_WANTED: set[str] = set()


def _init(wanted):
    global _WANTED
    _WANTED = wanted


def _canonical_chunk(texts):
    """(written, canonical) for the chunk's strings whose canonical form is wanted."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    hits, readable = [], 0
    for text in texts:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        readable += 1
        canon = Chem.MolToSmiles(mol)
        if canon in _WANTED:
            hits.append((text, canon))
    return hits, readable, len(texts)


def pair_molecules(pairs_file):
    """Canonical form of each pair's molecule, by pair position."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    pairs = [json.loads(line) for line in open(pairs_file, encoding="utf-8")
             if line.strip()]
    canon, unreadable = {}, []
    for i, pair in enumerate(pairs):
        mol = Chem.MolFromSmiles(pair["molecule"])
        if mol is None:
            unreadable.append(i)
        else:
            canon[i] = Chem.MolToSmiles(mol)
    return pairs, canon, unreadable


def scan_train(arrow_train, wanted, processes, batch_size, tokenizer_dir):
    """Which wanted molecules occur in train, and the task types they occur in."""
    import os as _os

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir or _os.environ["GPT2_TOKENIZER_DIR"])
    data = load_from_disk(arrow_train)
    rows = len(data)
    data = data.select_columns(["input_text", "output_ids", "task_type"])

    str_id: dict[str, int] = {}
    occ_str, occ_task = array("i"), array("i")
    task_id: dict[str, int] = {}
    task_rows = Counter()
    seen = 0
    for batch in data.iter(batch_size=batch_size):
        responses = tokenizer.batch_decode(batch["output_ids"], skip_special_tokens=True)
        for prompt, response, task in zip(batch["input_text"], responses,
                                          batch["task_type"]):
            seen += 1
            task_rows[task] += 1
            tid = task_id.setdefault(task, len(task_id))
            for side in (prompt, response):
                side = (side or "").strip()
                # The earlier overlap run's own test for a molecule-shaped side.
                if not side or " " in side:
                    continue
                sid = str_id.get(side)
                if sid is None:
                    sid = str_id[side] = len(str_id)
                occ_str.append(sid)
                occ_task.append(tid)
        if seen % (batch_size * 100) == 0:
            print(f"    read {seen:,} of {rows:,} records, "
                  f"{len(str_id):,} distinct molecule-shaped strings", flush=True)
    print(f"  train records read: {seen:,} (dataset says {rows:,})", flush=True)
    print(f"  molecule-shaped strings: {len(str_id):,} distinct", flush=True)

    written = list(str_id)
    chunks = [written[i:i + 20_000] for i in range(0, len(written), 20_000)]
    matched: dict[str, str] = {}
    readable = done = 0
    with Pool(processes, initializer=_init, initargs=(wanted,)) as pool:
        for i, (hits, ok, n) in enumerate(pool.imap_unordered(_canonical_chunk, chunks), 1):
            matched.update(hits)
            readable += ok
            done += n
            if i % 20 == 0:
                print(f"    canonicalised {done:,} of {len(written):,} "
                      f"-> {len(matched):,} written forms matched", flush=True)
    print(f"  canonicalised {done:,} strings, {readable:,} RDKit read, "
          f"{len(matched):,} written forms match a pair molecule", flush=True)

    matched_ids = {str_id[w]: matched[w] for w in matched}
    by_molecule = defaultdict(Counter)
    id_to_task = {v: k for k, v in task_id.items()}
    for sid, tid in zip(occ_str, occ_task):
        canon = matched_ids.get(sid)
        if canon is not None:
            by_molecule[canon][id_to_task[tid]] += 1
    return {
        "records_read": seen,
        "records_in_dataset": rows,
        "task_rows": dict(task_rows.most_common()),
        "molecule_shaped_strings": len(str_id),
        "molecule_shaped_rdkit_read": readable,
        "written_forms_matched": len(matched),
        "molecules_matched": sorted(by_molecule),
        "task_counts_by_molecule": {m: dict(c.most_common())
                                    for m, c in sorted(by_molecule.items())},
    }


def read_ranks(ranks_dir):
    """rank per (arch, size, direction, query), with duplicate files cross-checked."""
    frames: dict[tuple, dict[int, int]] = {}
    source: dict[tuple, str] = {}
    disagreements = []
    for path in sorted(glob.glob(os.path.join(ranks_dir, "*-ranks.tsv"))):
        with open(path, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(
                (line for line in fh if not line.startswith("#")), delimiter="\t")
                if r.get("arch") and r.get("arch") != "arch"]
        here = defaultdict(dict)
        for row in rows:
            here[(row["arch"], row["size"], row["direction"])][int(row["query"])] = \
                int(row["rank"])
        for key, ranks in here.items():
            if key in frames:
                if frames[key] != ranks:
                    disagreements.append({"frame": "/".join(key),
                                          "first": source[key], "second": path})
                continue
            frames[key] = ranks
            source[key] = path
    return frames, source, disagreements


def metrics(ranks):
    n = len(ranks)
    if not n:
        return {"queries": 0}
    values = list(ranks)
    return {
        "queries": n,
        "recall@1": sum(1 for r in values if r <= 1) / n,
        "recall@5": sum(1 for r in values if r <= 5) / n,
        "recall@10": sum(1 for r in values if r <= 10) / n,
        "mrr": sum(1.0 / r for r in values) / n,
        "median_rank": statistics.median(values),
    }


def stored_metrics(ranks_dir):
    """What the scoring runs wrote, to check the ranks are the same measurement."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ranks_dir, "retrieval-*.json"))):
        summary = json.load(open(path, encoding="utf-8"))
        for direction, result in summary.get("directions", {}).items():
            out[(summary["arch"], summary["size"], direction)] = result
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arrow-train", required=True)
    ap.add_argument("--pairs-file", required=True)
    ap.add_argument("--ranks-dir", required=True)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsv", required=True)
    a = ap.parse_args()

    print("== pairs ==", flush=True)
    pairs, canon, unreadable = pair_molecules(a.pairs_file)
    wanted = set(canon.values())
    print(f"  {len(pairs):,} pairs, {len(wanted):,} distinct canonical molecules, "
          f"{len(unreadable)} RDKit would not read", flush=True)

    # The rank checks come before the scan. They cost a second, the scan costs twelve
    # minutes, and a failed check after the scan throws the scan away -- which is what
    # happened on the first run of this.
    print("== ranks ==", flush=True)
    frames, source, disagreements = read_ranks(a.ranks_dir)
    if disagreements:
        raise SystemExit(f"rank files disagree for the same frame: {disagreements}")
    stored = stored_metrics(a.ranks_dir)
    mismatches = []
    for key, ranks in sorted(frames.items()):
        full = metrics([ranks[q] for q in sorted(ranks)])
        want = stored.get(key)
        if not want:
            mismatches.append({"frame": "/".join(key), "why": "no stored summary"})
            continue
        for field in ("recall@1", "recall@5", "recall@10", "mrr", "median_rank"):
            # The scoring runs wrote these rounded to four decimals, so the comparison is
            # at that precision. A tighter one fails on the rounding alone: bert large
            # molecule_to_text recomputes to 0.18289224 against a stored 0.1829.
            if abs(full[field] - want[field]) > 5e-5:
                mismatches.append({"frame": "/".join(key), "field": field,
                                   "recomputed": full[field], "stored": want[field]})
    if mismatches:
        raise SystemExit(f"the ranks do not reproduce the scoring summaries: {mismatches}")
    print(f"  {len(frames)} frames, all reproducing their scoring summary", flush=True)

    print("== train scan ==", flush=True)
    scan = scan_train(a.arrow_train, wanted, a.processes, a.batch_size, a.tokenizer_dir)
    in_train_mols = set(scan["molecules_matched"])
    in_train = sorted(i for i, c in canon.items() if c in in_train_mols)
    not_in_train = sorted(i for i, c in canon.items() if c not in in_train_mols)
    print(f"  pair positions whose molecule is in train: {len(in_train)} "
          f"({len(in_train_mols)} distinct molecules)", flush=True)
    print(f"  pair positions whose molecule is not: {len(not_in_train)}", flush=True)
    print(f"  pair positions RDKit would not read: {len(unreadable)}", flush=True)

    task_totals = Counter()
    for counts in scan["task_counts_by_molecule"].values():
        for task, n in counts.items():
            task_totals[task] += n
    molecules_by_task = Counter()
    for counts in scan["task_counts_by_molecule"].values():
        for task in counts:
            molecules_by_task[task] += 1
    in_molecule_tasks = sorted(
        m for m, counts in scan["task_counts_by_molecule"].items()
        if any(t in MOLECULE_TASKS for t in counts))
    print("== task types of the matched molecules ==", flush=True)
    for task, n in molecules_by_task.most_common():
        print(f"  {task}: {n} molecules, {task_totals[task]:,} occurrences", flush=True)
    print(f"  molecules occurring in {' or '.join(MOLECULE_TASKS)}: "
          f"{len(in_molecule_tasks)}", flush=True)

    subsets = (("molecule_in_train", in_train),
               ("molecule_not_in_train", not_in_train),
               ("molecule_rdkit_unreadable", unreadable),
               ("all", sorted(range(len(pairs)))))
    rows = []
    for key in sorted(frames):
        for name, positions in subsets:
            ranks = [frames[key][q] for q in positions if q in frames[key]]
            row = {"arch": key[0], "size": key[1], "direction": key[2], "subset": name}
            row.update(metrics(ranks))
            rows.append(row)

    answers = {
        "pairs": len(pairs),
        "pair_positions_molecule_in_train": in_train,
        "pair_positions_molecule_not_in_train_count": len(not_in_train),
        "pair_positions_molecule_rdkit_unreadable": unreadable,
        "distinct_canonical_molecules": len(wanted),
        "distinct_canonical_molecules_in_train": len(in_train_mols),
        "task_occurrences_total": dict(task_totals.most_common()),
        "molecules_per_task": dict(molecules_by_task.most_common()),
        "molecules_in_molecule_description_tasks": in_molecule_tasks,
        "train_scan": {k: v for k, v in scan.items() if k != "task_counts_by_molecule"},
        "task_counts_by_molecule": scan["task_counts_by_molecule"],
        "rank_file_per_frame": {"/".join(k): v for k, v in sorted(source.items())},
        "split_metrics": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(answers, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  wrote {a.out}", flush=True)

    columns = ("arch", "size", "direction", "subset", "queries", "recall@1", "recall@5",
               "recall@10", "mrr", "median_rank")
    os.makedirs(os.path.dirname(os.path.abspath(a.tsv)), exist_ok=True)
    with open(a.tsv, "w", encoding="utf-8") as fh:
        fh.write("#! report-tsv 1\n")
        fh.write("# model: several (columns: arch,size)\n")
        fh.write("# run: molecule_nat_lang retrieval on the SMolInstruct test split, "
                 "split by whether the pair's molecule occurs in train\n")
        fh.write(f"# source: pairs {os.path.abspath(a.pairs_file)}\n")
        fh.write(f"# source: ranks {os.path.abspath(a.ranks_dir)}\n")
        fh.write(f"# source: train split {os.path.abspath(a.arrow_train)}\n")
        fh.write(f"# generated: "
                 f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
                 f" by scripts/molnl_smoltest_train_overlap_split.py\n")
        fh.write(f"# note: subset molecule_in_train has {len(in_train)} of the 1,000 "
                 f"pair positions, from {len(in_train_mols)} distinct canonical "
                 f"molecules\n")
        fh.write(f"# note: molecule_rdkit_unreadable is the {len(unreadable)} positions "
                 "whose molecule RDKit will not read; they are not in either of the "
                 "other two subsets\n")
        fh.write("# note: nothing was scored again; the per-query ranks come from the "
                 "2026-11-05 and 2026-11-06 scoring runs\n")
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(
                f"{row[c]:.4f}" if isinstance(row[c], float) and c != "median_rank"
                else (f"{row[c]:.1f}" if c == "median_rank" else str(row[c]))
                for c in columns) + "\n")
    print(f"  wrote {a.tsv}: {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
