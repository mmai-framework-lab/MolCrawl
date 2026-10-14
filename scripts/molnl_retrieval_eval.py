#!/usr/bin/env python3
"""Retrieval between molecules and their descriptions, both directions.

The 2026-10-14 order asks how well a pretrained model can pick the description that
belongs to a molecule, and the molecule that belongs to a description, on the held-out
test pairs. Neither model was trained for retrieval -- there is no contrastive objective
and no projection head -- so the model is asked the only question it can answer: of these
candidates, which pairing does it find least surprising.

For each query the true counterpart and K-1 distractors are formatted into the same
template the pair evaluator uses, scored with the adapter's own likelihood (a causal
log-likelihood for GPT-2, a pseudo-log-likelihood for BERT), and ranked. Recall@1,
Recall@5 and MRR come out of the rank of the true one.

Two things this does not do. It does not compare BERT's numbers with GPT-2's as if they
were the same measurement -- a pseudo-log-likelihood and a causal log-likelihood are
different quantities, and only the ranks they induce are comparable. And it does not
decide whether the test pairs were in the pretraining corpus; see
molnl_downstream_contamination.py.

    python scripts/molnl_retrieval_eval.py --model-path <ckpt> --arch gpt2 --size small \
        --pairs-root <Mol-Instructions dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import time

from molcrawl.tasks.evaluation import _adapters  # noqa: F401  registers the adapters
from molcrawl.tasks.evaluation._base import ModelHandle
from molcrawl.tasks.evaluation._base.model_adapter import build_adapter

DIRECTIONS = ("molecule_to_text", "text_to_molecule")


def test_pairs(root, subset="molecular_description_generation"):
    """[(molecule, description)] for the rows Mol-Instructions marks as test."""
    from datasets import load_from_disk

    data = load_from_disk(os.path.join(root, subset))
    pairs = []
    for item in data:
        meta = item["metadata"]
        if not isinstance(meta, dict):
            meta = json.loads(str(meta).replace("'", '"'))
        if str(meta.get("split", "")).lower() == "test":
            pairs.append((item["input"], item["output"]))
    return pairs


def pools(n, k, rng):
    """For each query, the indices it ranks: itself first, then k-1 others."""
    out = []
    for i in range(n):
        others = [j for j in rng.sample(range(n), min(n, k + 5)) if j != i][:k - 1]
        out.append([i] + others)
    return out


def rank_of_true(scores):
    """1 if the true pairing scored highest. Ties count against it."""
    true_score = scores[0]
    return 1 + sum(1 for s in scores[1:] if s >= true_score)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--arch", required=True, choices=("gpt2", "bert"))
    ap.add_argument("--size", default=None)
    ap.add_argument("--label", default=None, help="name for the output files")
    ap.add_argument("--pairs-root", required=True)
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--modality", default="molecule_nat_lang")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--template", default="{caption}\n{smiles}")
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--queries", type=int, default=0, help="0 uses every test pair")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    label = a.label or f"{a.arch}_{a.size or 'model'}"

    pairs = test_pairs(a.pairs_root)
    if a.queries:
        pairs = pairs[:a.queries]
    rng = random.Random(a.seed)
    candidate_sets = pools(len(pairs), a.candidates, rng)
    print(f"  {len(pairs):,} queries x {a.candidates} candidates, template {a.template!r}")

    handle = ModelHandle(arch=a.arch, modality=a.modality, model_path=a.model_path,
                         tokenizer_path=a.tokenizer_path, size=a.size,
                         extras={"device": a.device})
    adapter = build_adapter(handle)
    adapter.load()

    results = {}
    rows = []
    try:
        for direction in DIRECTIONS:
            ranks = []
            started = time.time()
            for qi, pool in enumerate(candidate_sets):
                molecule, description = pairs[qi]
                if direction == "molecule_to_text":
                    texts = [a.template.format(caption=pairs[c][1], smiles=molecule)
                             for c in pool]
                else:
                    texts = [a.template.format(caption=description, smiles=pairs[c][0])
                             for c in pool]
                scores = []
                for start in range(0, len(texts), a.batch):
                    out = adapter.score_likelihood(texts[start:start + a.batch])
                    # LikelihoodOutput.log_likelihood is one value per sequence; the
                    # adapters return it as a tensor or an array depending on the arch.
                    chunk = out.log_likelihood
                    chunk = chunk.tolist() if hasattr(chunk, "tolist") else list(chunk)
                    scores.extend(float(v) for v in chunk)
                rank = rank_of_true(scores)
                ranks.append(rank)
                rows.append((label, a.arch, a.size or "", direction, qi, rank,
                             f"{scores[0]:.6f}", f"{max(scores):.6f}"))
                if (qi + 1) % 100 == 0:
                    rate = (qi + 1) / (time.time() - started)
                    print(f"    {direction}: {qi + 1:,}/{len(pairs):,} "
                          f"({rate:.2f} query/s, R@1 so far "
                          f"{sum(1 for r in ranks if r == 1) / len(ranks):.3f})")
            results[direction] = {
                "queries": len(ranks), "candidates": a.candidates,
                "recall@1": round(sum(1 for r in ranks if r == 1) / len(ranks), 4),
                "recall@5": round(sum(1 for r in ranks if r <= 5) / len(ranks), 4),
                "recall@10": round(sum(1 for r in ranks if r <= 10) / len(ranks), 4),
                "mrr": round(st.mean(1.0 / r for r in ranks), 4),
                "random_baseline_recall@1": round(1 / a.candidates, 4),
                "seconds": round(time.time() - started, 1),
            }
            print(f"  {direction}: {json.dumps(results[direction])}")
    finally:
        adapter.close()

    summary = {"label": label, "arch": a.arch, "size": a.size,
               "model_path": a.model_path, "template": a.template,
               "seed": a.seed, "directions": results}
    json.dump(summary, open(os.path.join(a.out_dir, f"retrieval-{label}.json"), "w"), indent=2)
    with open(os.path.join(a.out_dir, f"retrieval-{label}-ranks.tsv"), "w") as fh:
        fh.write("model\tarch\tsize\tdirection\tquery\trank\ttrue_score\tbest_score\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"  wrote retrieval-{label}.json and -ranks.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
