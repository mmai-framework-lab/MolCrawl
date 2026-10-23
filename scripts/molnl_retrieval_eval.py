#!/usr/bin/env python3
"""Retrieval between molecules and their descriptions, both directions.

The 2026-10-14 order asks how well a pretrained model can pick the description that
belongs to a molecule, and the molecule that belongs to a description, on the held-out
test pairs. Neither model was trained for retrieval -- there is no contrastive objective
and no projection head -- so the model is asked the only question it can answer: of these
candidates, which pairing does it find least surprising.

The 2026-10-23 judgment adds a second axis: the first measurements asked in a form the
models never saw in pretraining, and the two mismatches are separated here so each can be
read on its own.

  --wrap inst            writes the record the way the corpus does,
                         "<s>[INST] query [/INST]candidate<|endoftext|>", taken from
                         decoding the packed corpus rather than from the formatting code.
                         --wrap plain keeps the newline the first measurements used.
  --molecule-notation    the corpus is built from osunlp/SMolInstruct, which writes
                         molecules as SMILES; the evaluation pairs come from
                         zjunlp/Mol-Instructions, which publishes SELFIES. "smiles"
                         decodes each SELFIES and canonicalises it with RDKit. Only 9.0%
                         of the corpus's own SMILES are in that canonical form, so this
                         matches the notation, not the house style.

Three things decide whether that question is asked properly.

  the order      a causal model cannot condition on text that comes after. The query is
                 therefore written first and the candidate second, which means the two
                 directions use opposite orders: molecule then description, description
                 then molecule. Scoring a candidate that precedes its query would rank
                 candidates by their own unconditional likelihood.
  what is scored only the candidate's own token positions, so the score is
                 log p(candidate | query) per token. Averaging over the whole pair mixes
                 in the query's likelihood, which is the same for every candidate and
                 only adds noise to the ranking.
  the context    the adapters default to 512 tokens and truncate past it. 44 of these
                 1,000 pairs are longer than 512 and 10 are longer than 1,024, so the
                 context is set explicitly and what still got truncated is reported.

BERT's pseudo-log-likelihood and GPT-2's causal log-likelihood are different quantities;
only the ranks they induce are comparable, so no number here is compared across the two.
Whether the test pairs were pretrained on is a separate question --
scripts/molnl_downstream_pair_overlap.py.

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


def to_smiles(pairs):
    """Rewrite each pair's molecule as a canonical SMILES, dropping what will not decode.

    A pair whose molecule cannot be read is removed rather than carried as its SELFIES:
    the point of the run is that every molecule is in the corpus's notation.
    """
    import selfies as selfies_mod
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out, failed = [], 0
    for molecule, description in pairs:
        try:
            smiles = selfies_mod.decoder(molecule)
        except Exception:
            smiles = None
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            failed += 1
            continue
        out.append((Chem.MolToSmiles(mol), description))
    return out, failed


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


def candidate_span(adapter, prefix, whole, context):
    """The candidate's token positions inside the formatted pair, and the pair's length.

    The adapter's own encoder is used rather than a tokenizer of our own: the positions
    have to be the ones the scorer will see, including the special tokens it adds and the
    truncation it applies. The span is the formatted pair minus its longest common prefix
    with the query alone, which drops BERT's trailing separator token without having to
    know that it is there.
    """
    pre = adapter._encode(prefix)
    full = adapter._encode(whole, max_length=context)
    n = 0
    while n < len(pre) and n < len(full) and pre[n] == full[n]:
        n += 1
    return list(range(n, len(full))), len(full)


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
    ap.add_argument("--separator", default="\n", help="written between query and candidate")
    ap.add_argument("--wrap", default="plain", choices=("plain", "inst"),
                    help="plain joins with --separator; inst writes the corpus's own "
                         "<s>[INST] ... [/INST]...<|endoftext|> record")
    ap.add_argument("--molecule-notation", default="selfies", choices=("selfies", "smiles"),
                    help="selfies keeps the pairs as published; smiles decodes them and "
                         "canonicalises with RDKit, which is the notation the corpus uses")
    ap.add_argument("--bert-mlm-batch", type=int, default=256,
                    help="masked copies BERT scores in one forward; the adapter's own "
                         "default of 64 leaves a GB200 idle and the description->molecule "
                         "direction needs one copy per molecule token")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--queries", type=int, default=0,
                    help="how many queries to score; 0 scores every test pair. The "
                         "candidates are always drawn from all 1,000 pairs, so this does "
                         "not make a query cheaper -- it makes fewer of them")
    ap.add_argument("--queries-text-to-molecule", type=int, default=0,
                    help="overrides --queries for the description->molecule direction, "
                         "which for BERT costs one masked forward per molecule token and "
                         "is some 6x the other direction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    label = a.label or f"{a.arch}_{a.size or 'model'}"

    pairs = test_pairs(a.pairs_root)
    if a.molecule_notation == "smiles":
        pairs, failed = to_smiles(pairs)
        print(f"  SELFIES -> canonical SMILES: {len(pairs):,} pairs, {failed} undecodable")
    rng = random.Random(a.seed)
    # Pools over every pair whatever is scored: truncating the pair list first would
    # shrink the candidate pool with it, which once made a 20-query measurement look five
    # times faster per query than the real thing.
    candidate_sets = pools(len(pairs), a.candidates, rng)
    counts = {"molecule_to_text": a.queries or len(pairs),
              "text_to_molecule": (a.queries_text_to_molecule or a.queries or len(pairs))}
    print(f"  {len(pairs):,} pairs, {a.candidates} candidates, context {a.context}; "
          f"scoring {counts['molecule_to_text']:,} and {counts['text_to_molecule']:,} "
          f"queries in the two directions")

    handle = ModelHandle(arch=a.arch, modality=a.modality, model_path=a.model_path,
                         tokenizer_path=a.tokenizer_path, size=a.size,
                         extras={"device": a.device, "bert_mlm_batch": a.bert_mlm_batch})
    adapter = build_adapter(handle)
    adapter.load()

    summary = {"label": label, "arch": a.arch, "size": a.size,
               "model_path": a.model_path, "separator": a.separator,
               "wrap": a.wrap, "molecule_notation": a.molecule_notation,
               "context": a.context, "scored": "candidate positions only",
               "bert_mlm_batch": a.bert_mlm_batch, "seed": a.seed, "directions": {}}
    json_path = os.path.join(a.out_dir, f"retrieval-{label}.json")
    tsv_path = os.path.join(a.out_dir, f"retrieval-{label}-ranks.tsv")

    def write(results, rows):
        summary["directions"] = results
        json.dump(summary, open(json_path, "w"), indent=2)
        with open(tsv_path, "w") as fh:
            fh.write("model\tarch\tsize\tdirection\tquery\trank\ttrue_score\tbest_score\n")
            for row in rows:
                fh.write("\t".join(str(v) for v in row) + "\n")

    results = {}
    rows = []
    try:
        for direction in DIRECTIONS:
            ranks = []
            truncated = 0
            empty_span = 0
            started = time.time()
            for qi, pool in enumerate(candidate_sets[:counts[direction]]):
                molecule, description = pairs[qi]
                query = molecule if direction == "molecule_to_text" else description
                prefix = (f"<s>[INST] {query} [/INST]" if a.wrap == "inst"
                          else f"{query}{a.separator}")
                scores = []
                for c in pool:
                    candidate = pairs[c][1] if direction == "molecule_to_text" else pairs[c][0]
                    whole = (f"{prefix}{candidate}<|endoftext|>" if a.wrap == "inst"
                             else f"{prefix}{candidate}")
                    positions, length = candidate_span(adapter, prefix, whole, a.context)
                    if length >= a.context:
                        truncated += 1
                    if not positions:
                        # The query alone filled the context: nothing of the candidate
                        # is left to score, and a score of 0 would outrank every real
                        # one. Counted and given the worst finite score instead.
                        empty_span += 1
                        scores.append(float("-inf"))
                        continue
                    out = adapter.score_likelihood(
                        [whole], context_length=a.context, eval_position_indices=positions)
                    chunk = out.log_likelihood
                    chunk = chunk.tolist() if hasattr(chunk, "tolist") else list(chunk)
                    scores.append(float(chunk[0]))
                rank = rank_of_true(scores)
                ranks.append(rank)
                rows.append((label, a.arch, a.size or "", direction, qi, rank,
                             f"{scores[0]:.6f}", f"{max(scores):.6f}"))
                if (qi + 1) % 50 == 0:
                    rate = (qi + 1) / (time.time() - started)
                    print(f"    {direction}: {qi + 1:,}/{len(pairs):,} "
                          f"({rate:.2f} query/s, R@1 so far "
                          f"{sum(1 for r in ranks if r == 1) / len(ranks):.3f})", flush=True)
            results[direction] = {
                "queries": len(ranks), "pairs_drawn_from": len(pairs),
                "candidates": a.candidates,
                "recall@1": round(sum(1 for r in ranks if r == 1) / len(ranks), 4),
                "recall@5": round(sum(1 for r in ranks if r <= 5) / len(ranks), 4),
                "recall@10": round(sum(1 for r in ranks if r <= 10) / len(ranks), 4),
                "mrr": round(st.mean(1.0 / r for r in ranks), 4),
                "random_baseline_recall@1": round(1 / a.candidates, 4),
                "median_rank": st.median(ranks),
                "scorings_truncated": truncated,
                "scorings_with_no_candidate_left": empty_span,
                "scorings": len(ranks) * a.candidates,
                "seconds": round(time.time() - started, 1),
            }
            print(f"  {direction}: {json.dumps(results[direction])}", flush=True)
            write(results, rows)
    finally:
        adapter.close()

    write(results, rows)
    print(f"  wrote retrieval-{label}.json and -ranks.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
