"""OrganiX13 打ち切り前トークン長分布 + 128超/256超 の性状調査 (方針α: sagawa 単体).

Implements `instruction-organix13-length-survey.md` (v2).

- Read all 4 sagawa/OrganiX13 shards (train x2 + valid + test)
- Tokenize each SMILES with the SAME CompoundsTokenizer (vocab.txt, 612)
  used in training_ready — but WITHOUT truncation, so we see the FULL length
- Compute statistics + histogram
- For >128 / >256 SMILES: RDKit-based heuristic class breakdown + heavy atom
  count distribution + representative examples
- No 由来 breakdown (sagawa lacks source attribution)

Outputs → tmp/organix13_length_survey_20260707/
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
CACHE = REPO / "tmp" / "organix13_sagawa_cache" / "data"
BUNDLE = REPO / "tmp" / "organix13_length_survey_20260707"
BUNDLE.mkdir(parents=True, exist_ok=True)

SHARDS = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "valid": ["valid-00000-of-00001.parquet"],
    "test":  ["test-00000-of-00001.parquet"],
}

BINS = [(0, 32), (33, 64), (65, 128), (129, 256), (257, 512), (513, 1024), (1025, 10_000_000)]


def load_tokenizer():
    sys.path.insert(0, str(REPO))
    from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer
    # max_len 大きめに設定して truncation を実質無効化 (実 SMILES は最大でも数千 token)
    tok = CompoundsTokenizer("assets/molecules/vocab.txt", max_len=100_000)
    return tok


def token_len_for(tok, smiles: str) -> int:
    """Return token count for a SMILES without special tokens (raw content only)."""
    # SmilesTokenizer.encode with padding=max_length + truncation strips control,
    # but we want raw count. Use ._tokenize / tokenize under-the-hood.
    # Fall back to encode without padding.
    ids = tok.encode(smiles, padding=False, truncation=False, add_special_tokens=False)
    return len(ids)


def compute_distribution():
    tok = load_tokenizer()
    logger.info(f"Tokenizer: {type(tok).__name__}, vocab={tok.vocab_size}")

    import pyarrow.parquet as pq

    lens_by_split = {}
    long_smiles = {"gt128": [], "gt256": []}  # store (length, split, smiles)

    for split, files in SHARDS.items():
        all_lens = []
        for f in files:
            path = CACHE / f
            logger.info(f"[{split}] read {path.name}")
            tbl = pq.read_table(str(path), columns=["smiles"])
            smiles_arr = tbl.column("smiles").to_pylist()
            t0 = time.time()
            lens = np.zeros(len(smiles_arr), dtype=np.int32)
            for i, s in enumerate(smiles_arr):
                L = token_len_for(tok, s)
                lens[i] = L
                if L > 256 and len(long_smiles["gt256"]) < 200:
                    long_smiles["gt256"].append((int(L), split, s))
                elif 128 < L <= 256 and len(long_smiles["gt128"]) < 200:
                    long_smiles["gt128"].append((int(L), split, s))
                if i % 200_000 == 0 and i > 0:
                    logger.info(f"  processed {i:,}/{len(smiles_arr):,} in {time.time()-t0:.1f}s")
            all_lens.append(lens)
            logger.info(f"  {f}: {len(smiles_arr):,} rows in {time.time()-t0:.1f}s")
        lens_by_split[split] = np.concatenate(all_lens)
        logger.info(f"[{split}] total {len(lens_by_split[split]):,} rows")

    return lens_by_split, long_smiles


def summarise(lens, label: str) -> dict:
    stat = {
        "label": label,
        "n": int(len(lens)),
        "mean": float(lens.mean()),
        "median": float(np.median(lens)),
        "p95": float(np.percentile(lens, 95)),
        "p99": float(np.percentile(lens, 99)),
        "p99.9": float(np.percentile(lens, 99.9)),
        "max": int(lens.max()),
        "min": int(lens.min()),
    }
    histogram = {}
    for lo, hi in BINS:
        bin_label = f"{lo}-{hi if hi != 10_000_000 else '1024+'}"
        count = int(((lens >= lo) & (lens <= hi)).sum())
        pct = 100.0 * count / len(lens)
        histogram[bin_label] = {"count": count, "pct": pct}
    stat["histogram"] = histogram
    return stat


def heuristic_class(smi: str) -> tuple[str, dict]:
    """RDKit-based heuristic molecule class + metadata.

    Categories:
      - macrocycle: any ring size >= 12
      - polycyclic_aromatic (縮環系 aromatic): >= 3 fused aromatic rings
      - conjugated_large: has >= 6 conjugated double bonds (系 e.g. OPV/CEP)
      - sugar_like: multiple hydroxyl on ring (OH count >= 3 and any pyranose/furanose ring)
      - heterocycle: has any heterocyclic ring
      - other
    Returns (class_label, {'heavy_atoms': int, 'ring_count': int, 'max_ring_size': int, 'aromatic_rings': int, 'oh_count': int})
    """
    from rdkit import Chem

    m = Chem.MolFromSmiles(smi)
    if m is None:
        return "invalid_smiles", {"heavy_atoms": 0}

    n_heavy = m.GetNumHeavyAtoms()
    ri = m.GetRingInfo()
    ring_sizes = [len(r) for r in ri.AtomRings()]
    max_ring = max(ring_sizes) if ring_sizes else 0
    n_rings = len(ring_sizes)

    aromatic_rings = 0
    for ring in ri.AtomRings():
        if all(m.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings += 1

    # OH: rough proxy for sugar
    oh_count = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "O" and any(nb.GetSymbol() == "H" for nb in a.GetNeighbors()))
    # RDKit strips H by default; use SMARTS
    oh_count = len(m.GetSubstructMatches(Chem.MolFromSmarts("[OX2H1]")))

    # Conjugated double bonds
    n_conj_dbl = 0
    for b in m.GetBonds():
        if b.GetIsConjugated() and b.GetBondType() == Chem.BondType.DOUBLE:
            n_conj_dbl += 1

    meta = {
        "heavy_atoms": n_heavy,
        "ring_count": n_rings,
        "max_ring_size": max_ring,
        "aromatic_rings": aromatic_rings,
        "oh_count": oh_count,
        "n_conjugated_double": n_conj_dbl,
    }

    # priority ordering
    if max_ring >= 12:
        return "macrocycle", meta
    if aromatic_rings >= 3:
        return "polycyclic_aromatic", meta
    if n_conj_dbl >= 6:
        return "conjugated_large", meta
    if oh_count >= 3 and any(6 <= s or s == 5 for s in ring_sizes) and n_rings >= 1:
        return "sugar_like", meta
    if n_rings >= 1 and any("N" in Chem.MolToSmiles(m) or "O" in Chem.MolToSmiles(m) for _ in [0]):
        return "heterocycle_containing", meta
    return "other", meta


def analyse_long(long_smiles: dict) -> dict:
    out = {}
    for tag in ("gt128", "gt256"):
        entries = long_smiles[tag]
        if not entries:
            out[tag] = {"n": 0, "note": "no samples"}
            continue
        classes = {}
        heavy_atoms = []
        examples = []  # (length, class, smiles, meta)
        for L, split, smi in entries:
            cls, meta = heuristic_class(smi)
            classes[cls] = classes.get(cls, 0) + 1
            heavy_atoms.append(meta["heavy_atoms"])
            examples.append({"length": L, "split": split, "smiles": smi, "class": cls, "meta": meta})
        # sort examples by class (representative from each)
        by_class = {}
        for ex in examples:
            by_class.setdefault(ex["class"], []).append(ex)
        rep = []
        for _cls, exs in by_class.items():
            # take shortest, median-ish, longest
            exs_sorted = sorted(exs, key=lambda x: x["length"])
            picks = [exs_sorted[0], exs_sorted[len(exs_sorted) // 2], exs_sorted[-1]]
            for p in picks:
                if p not in rep:
                    rep.append(p)
        rep = rep[:20]  # cap
        ha = np.array(heavy_atoms)
        out[tag] = {
            "n_sampled": len(entries),
            "class_counts": classes,
            "class_pct": {k: 100 * v / len(entries) for k, v in classes.items()},
            "heavy_atoms": {
                "mean": float(ha.mean()),
                "median": float(np.median(ha)),
                "p95": float(np.percentile(ha, 95)),
                "max": int(ha.max()),
                "min": int(ha.min()),
            },
            "examples": rep,
        }
    return out


def main() -> int:
    t0 = time.time()
    lens_by_split, long_smiles = compute_distribution()

    stats = {sp: summarise(lens_by_split[sp], sp) for sp in lens_by_split}
    all_lens = np.concatenate(list(lens_by_split.values()))
    stats["all"] = summarise(all_lens, "all")

    # 128超・256超 の分析
    logger.info("Analysing long-tail SMILES (>128 and >256) ...")
    tail_analysis = analyse_long(long_smiles)

    # Save
    out = {"stats_per_split": stats, "tail_analysis": tail_analysis, "elapsed_sec": time.time() - t0}
    (BUNDLE / "length_distribution.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info(f"[done] wrote {BUNDLE}/length_distribution.json")

    # CSV: histogram bin table
    with (BUNDLE / "length_distribution.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "bin", "count", "pct", "n_total", "mean", "median", "p95", "p99", "p99.9", "max"])
        for sp, s in stats.items():
            for bin_label, sub in s["histogram"].items():
                w.writerow([sp, bin_label, sub["count"], f"{sub['pct']:.4f}",
                            s["n"], f"{s['mean']:.2f}", s["median"],
                            s["p95"], s["p99"], s["p99.9"], s["max"]])
    logger.info(f"[done] wrote {BUNDLE}/length_distribution.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
