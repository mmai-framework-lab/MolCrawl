"""What a SMILES corpus loses at each filter the ZINC-22 order specifies.

The order's §3 lists eight steps, two of which are not decided by the existing
compounds pipeline: "drop what RDKit cannot parse", "drop over 256 tokens" and
"drop ion structures" need numbers before a definition can be chosen, and the
tokenizer step needs to be checked against a corpus wider than organix13.

Four readings of "ion structure" are counted separately rather than one being
assumed, because they do not select the same molecules and the difference is
large. A nitro group is written N+(=O)[O-]; by formal charge it is an ion, by net
charge it is not, and it is an ordinary group on a neutral molecule.

  charged_atom   some atom carries a non-zero formal charge
  multi_fragment the SMILES holds a '.', so it is a salt or a mixture
  net_charge     the formal charges do not sum to zero
  either         charged_atom or multi_fragment

The tokenizer is the one the corpus already uses -- WordPiece over a SMILES regex
with a 612-entry vocabulary, not sentencepiece -- and unknown tokens are counted
rather than assumed absent: a vocabulary fitted on one corpus meets characters it
has not seen in a wider one, and those become [UNK] silently.

Reads in batches; nothing is held whole.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from multiprocessing import Pool

MAX_TOKENS = 256
_TOK = None


def _tokenizer(vocab, max_len):
    global _TOK
    if _TOK is None:
        from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer
        _TOK = CompoundsTokenizer(vocab, max_len)
    return _TOK


def _profile(job):
    """One batch: counts only, so the parent never sees the molecules."""
    smiles, vocab, max_len = job
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    tok = _tokenizer(vocab, max_len)
    inner = getattr(tok, "tokenizer", tok)
    unk_id = getattr(inner, "unk_token_id", None)
    c = Counter()
    lengths = Counter()
    for smi in smiles:
        c["total"] += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            c["unparsed"] += 1
            continue
        c["parsed"] += 1
        charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
        if any(charges):
            c["charged_atom"] += 1
        if "." in smi:
            c["multi_fragment"] += 1
        if sum(charges) != 0:
            c["net_charge"] += 1
        if any(charges) or "." in smi:
            c["either"] += 1
        ids = inner.encode(smi, add_special_tokens=False)
        n = len(ids)
        lengths[min(n // 16 * 16, 512)] += 1
        if n > MAX_TOKENS:
            c["over_256_tokens"] += 1
        if unk_id is not None and unk_id in ids:
            c["has_unknown_token"] += 1
            c["unknown_tokens"] += sum(1 for i in ids if i == unk_id)
        c["tokens"] += n
    return c, lengths


def read_parquet(path, column, size, limit):
    import pyarrow.parquet as pq

    seen = 0
    for rb in pq.ParquetFile(path).iter_batches(batch_size=size, columns=[column]):
        vals = [v for v in rb.column(column).to_pylist() if v]
        if limit and seen + len(vals) > limit:
            vals = vals[: limit - seen]
        seen += len(vals)
        if vals:
            yield vals
        if limit and seen >= limit:
            return


def read_smi_gz(paths, size, limit):
    buf, seen = [], 0
    for p in paths:
        with gzip.open(p, "rt", errors="replace") as fh:
            for line in fh:
                s = line.split("\t", 1)[0].strip()
                if not s:
                    continue
                buf.append(s)
                seen += 1
                if len(buf) >= size:
                    yield buf
                    buf = []
                if limit and seen >= limit:
                    if buf:
                        yield buf
                    return
    if buf:
        yield buf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet")
    ap.add_argument("--column", default="smiles")
    ap.add_argument("--smi-gz", nargs="*", default=[])
    ap.add_argument("--vocab", default="assets/molecules/vocab.txt")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=20000)
    ap.add_argument("--processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.parquet:
        src = read_parquet(args.parquet, args.column, args.batch_size, args.limit)
    elif args.smi_gz:
        src = read_smi_gz(args.smi_gz, args.batch_size, args.limit)
    else:
        raise SystemExit("give --parquet or --smi-gz")

    total, lengths = Counter(), Counter()
    with Pool(args.processes) as pool:
        for c, lg in pool.imap_unordered(_profile, ((b, args.vocab, args.max_length) for b in src)):
            total.update(c)
            lengths.update(lg)

    n = total["total"] or 1
    print(f"=== {args.label or args.parquet or 'corpus'}")
    print(f"  {'分子数':<28}{total['total']:,}")
    for k, jp in (("unparsed", "RDKit で解析できない"), ("over_256_tokens", "256 トークン超"),
                  ("charged_atom", "電荷を持つ原子あり"), ("multi_fragment", "複数断片（. を含む）"),
                  ("net_charge", "正味電荷が 0 でない"), ("either", "電荷あり または 複数断片"),
                  ("has_unknown_token", "未知トークンを含む")):
        v = total[k]
        print(f"  {jp:<28}{v:,} ({100 * v / n:.3f} %)")
    if total["tokens"]:
        print(f"  {'総トークン':<28}{total['tokens']:,}")
        print(f"  {'1 分子あたり':<28}{total['tokens'] / n:.2f}")
        print(f"  {'未知トークン総数':<28}{total['unknown_tokens']:,} "
              f"({100 * total['unknown_tokens'] / total['tokens']:.5f} %)")
    print("  トークン長の分布（16 刻み、上位）:")
    for b, v in sorted(lengths.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {b:>4}-{b + 15:<4} {v:>12,} ({100 * v / n:.2f} %)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"label": args.label, "counts": dict(total),
                       "length_histogram": {str(k): v for k, v in sorted(lengths.items())}}, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
