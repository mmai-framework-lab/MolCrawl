"""Count the molecules actually in a packed corpus, split by split.

A packed corpus stores fixed blocks, not molecules, so its molecule count is only
visible as the number of separators it contains. Counting them per split matters
because a count taken from the train split alone is 80 per cent of the corpus and
looks exactly like a corpus that lost 20 per cent of its molecules.

Reports the token total beside it, because the two together say whether anything was
dropped: packed tokens should equal the source's tokens plus one separator per
molecule, less the tail of each shard that did not fill a block.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from datasets import load_from_disk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="DatasetDict directory")
    ap.add_argument("--sep-id", type=int, required=True)
    ap.add_argument("--report")
    args = ap.parse_args(argv)

    ds = load_from_disk(args.dataset)
    out = {}
    for name in ds:
        split = ds[name]
        seps = tokens = 0
        for batch in split.with_format("numpy").iter(batch_size=2048):
            arr = batch["input_ids"]
            flat = np.concatenate([np.asarray(a) for a in arr]) if isinstance(arr, list) \
                else np.asarray(arr).reshape(-1)
            tokens += flat.size
            seps += int((flat == args.sep_id).sum())
        out[name] = {"sequences": split.num_rows, "tokens": tokens, "separators": seps}
        print(f"  {name:6s} 系列 {split.num_rows:,}  トークン {tokens:,}  "
              f"[SEP] {seps:,}", flush=True)

    tot_seq = sum(v["sequences"] for v in out.values())
    tot_tok = sum(v["tokens"] for v in out.values())
    tot_sep = sum(v["separators"] for v in out.values())
    print(f"\n合計 系列 {tot_seq:,}  トークン {tot_tok:,}  [SEP] {tot_sep:,}")
    print(f"→ この corpus に入っている分子は {tot_sep:,} である")
    if tot_sep:
        print(f"→ 1 分子あたり {(tot_tok - tot_sep) / tot_sep:.2f} トークン（[SEP] を除く）")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"dataset": os.path.abspath(args.dataset), "splits": out,
                       "molecules": tot_sep, "tokens": tot_tok,
                       "sequences": tot_seq}, fh, ensure_ascii=False, indent=2)
        print(f"報告 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
