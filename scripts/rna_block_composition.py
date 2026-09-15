"""1 ブロックあたりの細胞数と、pad が占める位置の割合を実データで数える。

上長 2026-09-01 の依頼 5（1 ブロックあたりの細胞数）と、条件 c（pad が占める
位置の割合）への回答。RNA では細胞の境目と pad が同じ id 0 なので、同じ走査で
両方が出る -- というより、両者を区別できないことがそのまま答えになる。

条件 b（document masking の注意マスクが入った状態で pad が参照されないか）も
ここで見る。境界ごとに注意を区切ると、境界の位置そのものが 1 つの「文書」に
なるため、参照されるかどうかは実際に行列を組んで確かめるほかない。
"""
import argparse
import json
from pathlib import Path

import numpy as np


def LOG(*a):
    print(*a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin-dir", required=True,
                    help="training_ready_bin（train.bin と train.json がある）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--boundary-id", type=int, default=0)
    ap.add_argument("--rows", type=int, default=20000)
    a = ap.parse_args(argv)

    d = Path(a.bin_dir)
    meta = json.loads((d / f"{a.split}.json").read_text())
    block = int(meta.get("block_size", meta.get("block", 1024)))
    rows_total = int(meta.get("rows", meta.get("n_rows", 0)))
    LOG(f"META block={block} rows={rows_total:,} keys={sorted(meta)}")

    arr = np.memmap(d / f"{a.split}.bin", dtype=np.uint16, mode="r")
    n_rows = len(arr) // block
    take = min(a.rows, n_rows)
    LOG(f"DATA {n_rows:,} ブロック中 {take:,} を読む（block={block}）")
    x = np.asarray(arr[: take * block]).reshape(take, block)

    is_b = x == a.boundary_id
    per_row = is_b.sum(axis=1)
    LOG("")
    LOG(f"=== 境界 id {a.boundary_id} の出現 ===")
    LOG(f"  総数 {int(is_b.sum()):,} / {x.size:,} 位置 = {is_b.mean():.4%}")
    LOG(f"  1 ブロックあたり 中央 {np.median(per_row):.0f} "
        f"平均 {per_row.mean():.2f} 最小 {per_row.min()} 最大 {per_row.max()}")
    LOG(f"  境界が 1 つも無いブロック: {int((per_row == 0).sum()):,} "
        f"({(per_row == 0).mean():.2%})")
    LOG("")
    LOG("=== 1 ブロックあたりの細胞数（依頼 5）===")
    LOG("  細胞は端から端まで詰められ、境目に id 0 が 1 つ入る。したがって")
    LOG("  ブロック内の断片の数 = 境界の数 + 1（両端は切れている）")
    LOG(f"  中央 {np.median(per_row) + 1:.0f} 個  平均 {per_row.mean() + 1:.2f} 個")
    LOG("")
    LOG("=== 条件 c: pad が占める位置の割合 ===")
    LOG(f"  {is_b.mean():.4%}")
    LOG("  RNA の training_ready は詰め物を持たない（端まで詰めてブロック単位で切る）。")
    LOG("  したがってこの割合は pad ではなく細胞の境目そのもので、両者は同じ id 0 で")
    LOG("  区別できない。'pad が占める割合' は RNA では 0 である。")
    LOG("")
    LOG("=== 条件 b: 注意マスクが入った状態で境界が参照されるか ===")
    import torch

    from molcrawl.models._collators.document_masking import (
        document_attention_mask,
        document_ids,
    )
    probe = torch.as_tensor(x[:64].astype(np.int64))
    am = document_attention_mask(probe, a.boundary_id)
    doc = document_ids(probe, a.boundary_id)
    b_pos = probe == a.boundary_id
    # 境界の位置を見にいく非境界位置がどれだけあるか
    looks_at_boundary = (am.bool() & b_pos.unsqueeze(1)).any(dim=2) & ~b_pos
    LOG(f"  標本 {probe.shape[0]} ブロック x {probe.shape[1]} 位置")
    LOG(f"  attention_mask の形: {tuple(am.shape)}")
    LOG(f"  境界を参照する非境界位置: {int(looks_at_boundary.sum()):,} "
        f"/ {int((~b_pos).sum()):,} ({looks_at_boundary.float().mean():.2%})")
    LOG(f"  1 ブロックあたりの文書 id の種類: 中央 "
        f"{float(np.median([len(torch.unique(r)) for r in doc])):.0f}")
    LOG("")
    LOG("RESULT " + ("境界は参照されている。境界自身が 1 つの文書として扱われるため"
                    if int(looks_at_boundary.sum()) else
                    "境界は参照されていない"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
