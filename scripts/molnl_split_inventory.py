#!/usr/bin/env python3
"""What is in each molecule_nat_lang split, and which of it any run has read.

The 2026-11-04 order asks where the 48,125 records outside train went, what the residual
is, and whether the same retrieval evaluation could have been built from the test split
instead of a second distribution. Three things have to be kept apart to answer that, and
the earlier reports did not keep them apart:

  records   one SMolInstruct sample, which is what arrow_splits/*.arrow holds and what
            corpus-size-2026-10-27.tsv counts
  windows   1024 tokens of the packed corpus, which is what training_ready_hf_dataset*
            holds and what a run's steps and eval rows are counted in. A molnl record
            averages 99.7 tokens, so one window carries about ten of them
  tokens    windows x 1024, exactly, because the packer drops the final partial window

Mixing the first two is what makes the residual look like a split: 1,631 is a count of
windows and 48,125 a count of records, and subtracting one from the other names nothing.

Nothing is scored here and the corpus is not scanned record by record. Row counts come
from the arrow metadata; the token totals are summed over the one column that holds them.

    python scripts/molnl_split_inventory.py --arrow-root <arrow_splits> \
        --packed <training_ready_hf_dataset_shuffled> --out-json <json> --out-tsv <tsv>
"""

from __future__ import annotations

import argparse
import csv
import json
import os

WINDOW = 1024

# What each split is read by, as of 2026-11-04. Written here rather than derived, because
# "nothing reads it" is a statement about the configs and the run inventory, not about the
# files: every pretrain config under configs/molecule_nat_lang names the packed train for
# training and the packed valid for eval, and none of them names test.
USE = {
    "train": "事前学習（packed train を全窓）",
    "valid": "学習中の eval（packed valid を全窓）",
    "test": "使っていない",
}


def rows_and_columns(path):
    """Row count and column names of a save_to_disk directory, from its metadata."""
    from datasets import load_from_disk

    d = load_from_disk(path)
    return len(d), list(d.features), d


def token_totals(dataset, columns=("input_ids", "output_ids")):
    """Token totals per column.

    A record is packed as `<s>[INST] input [/INST]output<|endoftext|>`, so the packed
    stream holds the input, the output and a fixed number of marker tokens per record.
    Summing input_ids alone and comparing it with the packed total is meaningless -- it
    was the first thing tried here and it came out negative by 128M tokens.
    """
    out = {}
    have = [c for c in columns if c in dataset.features]
    if not have:
        return None
    for batch in dataset.with_format("numpy").iter(batch_size=10_000):
        for c in have:
            out[c] = out.get(c, 0) + sum(len(x) for x in batch[c])
    return out


def task_mix(dataset, limit=20):
    """How the records of a split divide by task, when it says what task each one is."""
    for key in ("task_type", "task", "metadata"):
        if key in dataset.features:
            counts = {}
            # Batched, because train has 3.3M rows and a row-at-a-time loop over it is
            # minutes of nothing.
            for batch in dataset.select_columns([key]).with_format("numpy").iter(
                    batch_size=20_000):
                for v in batch[key]:
                    v = str(v)
                    counts[v] = counts.get(v, 0) + 1
            return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:limit])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arrow-root", required=True,
                    help="the directory holding train.arrow / valid.arrow / test.arrow")
    ap.add_argument("--packed", required=True,
                    help="the packed training_ready_hf_dataset the runs read")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--sample-record", type=int, default=2,
                    help="print this many test records, to show whether a molecule and "
                         "its description can be read off one record")
    ap.add_argument("--out-json")
    ap.add_argument("--out-tsv")
    a = ap.parse_args()

    out = {"window": WINDOW, "splits": {}}
    for name in a.splits.split(","):
        rec_path = os.path.join(a.arrow_root, f"{name}.arrow")
        pack_path = os.path.join(a.packed, name)
        entry = {"records_path": rec_path, "packed_path": pack_path}

        n_rec, cols, ds_rec = rows_and_columns(rec_path)
        entry.update(records=n_rec, record_columns=cols)
        entry["record_tokens"] = token_totals(ds_rec)
        entry["task_mix"] = task_mix(ds_rec)
        print(f"  {name}: records {n_rec:,}  columns {cols}", flush=True)
        print(f"  {name}: task mix {entry['task_mix']}", flush=True)

        if os.path.isdir(pack_path):
            n_win, pcols, ds_pack = rows_and_columns(pack_path)
            entry.update(windows=n_win, packed_columns=pcols,
                         packed_tokens=n_win * WINDOW)
            lengths = {len(r["input_ids"]) for r in ds_pack.select(range(min(50, n_win)))}
            entry["packed_lengths_sampled"] = sorted(lengths)
            if entry["record_tokens"] is not None:
                # Solve for the marker tokens per record: the packed total is the inputs
                # plus the outputs plus k markers per record, minus the final partial
                # window the packer throws away. k is a small integer and the remainder
                # has to land in [0, 1023], which is what makes this a check and not a
                # fit -- only one k can satisfy both.
                body = sum(entry["record_tokens"].values())
                entry["packing_solution"] = next(
                    ({"markers_per_record": k,
                      "tokens_dropped_by_packing": body + k * n_rec - n_win * WINDOW}
                     for k in range(0, 33)
                     if 0 <= body + k * n_rec - n_win * WINDOW < WINDOW), None)
            print(f"  {name}: windows {n_win:,} = {n_win * WINDOW:,} tokens; "
                  f"record tokens {entry['record_tokens']}; "
                  f"packing {entry.get('packing_solution')}", flush=True)
        else:
            print(f"  {name}: no packed split at {pack_path}", flush=True)

        if name == "test":
            entry["sample"] = [
                {k: (str(ds_rec[i][k])[:200]) for k in cols}
                for i in range(min(a.sample_record, n_rec))]
            print(f"  test sample keys: {list(entry['sample'][0])}", flush=True)
        out["splits"][name] = entry

    if a.out_json:
        json.dump(out, open(a.out_json, "w"), ensure_ascii=False, indent=1)
        print(f"  wrote {a.out_json}")
    if a.out_tsv:
        with open(a.out_tsv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter="\t")
            # The four columns the order asks for, in its order. The two after them are
            # added because the record count and the window count are what get mixed up:
            # a run's eval rows are windows, and a split's size is records.
            w.writerow(["分割名", "レコード数", "置き場", "いまの用途",
                        "窓数（1024 トークン）", "packed の置き場"])
            for name, e in out["splits"].items():
                w.writerow([name, e["records"], e["records_path"], USE.get(name, ""),
                            e.get("windows", ""), e.get("packed_path", "")])
        print(f"  wrote {a.out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
