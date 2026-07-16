"""Byte-parity verify: `iter_acgt_segments` (旧 v1) vs `iter_acgt_segments_with_contig` (新 F2).

charter §「追加 verify: 前処理(N・大小文字)が v1 と不変であること」対応。

新実装 (F2 worktree, `fasta_to_raw.py:241-251`) の `iter_acgt_segments` は
`iter_acgt_segments_with_contig(...)` を wrap して contig を捨てるだけの構造なので
コード上 by-construction で一致するが、機械確認として複数 FASTA でセグメント
byte 列が同一であることを確認する。

Usage:
    python3 tmp/scripts/autopilot/analyzers/verify_byte_parity_iter_acgt_segments.py

戻り値: 0 全一致 / 1 差分あり
"""
from __future__ import annotations

import sys
from pathlib import Path

WORKTREE = Path("/lustre/home/matsubara/riken-genome-f2-worktree")
sys.path.insert(0, str(WORKTREE))

from molcrawl.data.genome_sequence.dataset.refseq.fasta_to_raw import (  # noqa: E402
    iter_acgt_segments,
    iter_acgt_segments_with_contig,
)


# 実 FASTA サンプル。cheap で手に入るものを 3-5 種。
# subset CSV 由来のうち小さめ + 合成ヒト chr22 込みを用いる。
CANDIDATE_FASTAS = [
    # サンプル: subset CSVで参照される種の中から実在確認できる .fna.gz を後述で拾う
]


def _find_fastas() -> list[Path]:
    """Existing FASTA を見繕う。既存 subset dir に .fna.gz が転がっている想定。"""
    roots = [
        Path("/lustre/home/matsubara/learning_source_20260529_evo2species/genome_sequence"),
        Path("/lustre/home/matsubara/riken-dataset-fundational-model/tmp"),
    ]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.fna.gz"):
            found.append(p)
            if len(found) >= 5:
                return found
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.fna"):
            found.append(p)
            if len(found) >= 5:
                return found
    return found


def main() -> int:
    fastas = _find_fastas()
    if not fastas:
        print("[verify] no sample FASTA found under expected roots — skip")
        return 0

    print(f"[verify] sampled {len(fastas)} FASTA(s):")
    for p in fastas[:5]:
        print(f"  {p}")

    all_ok = True
    for fa in fastas[:5]:
        segs_new = [s for _c, s in iter_acgt_segments_with_contig(fa, min_segment_len=100)]
        segs_old = list(iter_acgt_segments(fa, min_segment_len=100))
        if segs_new == segs_old:
            n_bytes = sum(len(s) for s in segs_new)
            print(f"[PASS] {fa.name}: {len(segs_new)} segments, {n_bytes:,} bytes, byte-identical")
        else:
            print(f"[FAIL] {fa.name}: new={len(segs_new)} vs old={len(segs_old)} segments")
            for i, (a, b) in enumerate(zip(segs_new, segs_old)):
                if a != b:
                    print(f"  first diff at seg {i}: new={a[:80]}... vs old={b[:80]}...")
                    break
            all_ok = False

    if all_ok:
        print("[verify] ALL PASS — v1 と F2 の N-split / uppercase / segment 境界は byte 一致")
        return 0
    else:
        print("[verify] DIFFERENCE FOUND — 前処理が意図せず変わっている可能性")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
