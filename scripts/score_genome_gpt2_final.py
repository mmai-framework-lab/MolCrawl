"""Final scoring for a finished genome GPT-2 run.

The campaign's second correlation takes each subset's *margin* -- how far the
run got below the loss a context-free predictor pays -- and asks whether it
explains the gap between ``global_random`` and ``eukaryote_matched``. A margin
is a difference, so it is only meaningful when both of its terms are measured on
the same positions. Until now they were not:

* the degenerate baseline came from the **first 10,000 valid rows**, and
* ``best_val`` came from the trainer's own evaluation, which draws
  ``batch_size`` x ``eval_iters`` = 320 x 200 = 64,000 rows **at random with
  replacement** from all 50,000, and keeps the smallest value any evaluation
  produced.

Neither difference is negligible. The baseline alone moves by -0.0141 to +0.0123
per subset between those two ranges, past the 0.0011 run-to-run noise in 18 of
21 subsets, and a minimum over noisy draws is biased low by construction. Mixed
into a correlation's x-axis, that is indistinguishable from a real difference in
how well the corpus was learned.

This scores the adopted checkpoint over every valid row **in order**, once, and
computes the baseline in the same pass from the labels at exactly the positions
that carried loss -- so the two terms cannot drift apart. It is the GPT-2
counterpart of ``score_genome_bert_final.py`` and reports the same fields.

There are no seeds here. Causal language modelling scores every position of
every row, so the pass is deterministic and one run is the answer, where the
BERT side has to average over masking draws.
"""
import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from molcrawl.models._collators import (
    ambiguous_tokens_for_modality,
    mask_ambiguous_targets_for_clm,
    resolve_ambiguous_token_ids,
)
from molcrawl.models._collators.ambiguity_aware_collator import IGNORE_INDEX
from molcrawl.models._representations import strip_compile_prefix
from molcrawl.models.gpt2.model import GPT, GPTConfig


def _entropy(counts):
    """Cross-entropy paid by predicting the marginal of these very labels."""
    total = sum(counts.values())
    if not total:
        return None
    return -sum((c / total) * math.log(c / total) for c in counts.values() if c)


def load_adopted(run_dir, device):
    """nanoGPT rewrites ckpt.pt whenever validation improves, so that file is
    the run's own selection -- not its newest checkpoint, which the 21 genome
    runs never agree with."""
    path = os.path.join(run_dir, "ckpt.pt")
    if not os.path.exists(path):
        raise SystemExit(f"{run_dir}: holds no ckpt.pt to score")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**dict(ck["model_args"])))
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    model.to(device).eval()
    best = ck.get("best_val_loss")
    return model, ck, (float(best) if best is not None else None)


def score(model, rows, ambiguous_ids, batch_size, device):
    """One ordered pass: summed loss over scored positions, and the label
    counts at those same positions."""
    total_loss, n_scored, n_ambiguous = 0.0, 0, 0
    counts = Counter()
    with torch.no_grad():
        for lo in range(0, len(rows), batch_size):
            batch = torch.tensor(rows[lo:lo + batch_size]["input_ids"],
                                 dtype=torch.long, device=device)
            x, y = batch[:, :-1], batch[:, 1:]
            y = mask_ambiguous_targets_for_clm(y, ambiguous_ids)
            logits, _ = model(x, y)
            flat_y = y.reshape(-1)
            keep = flat_y != IGNORE_INDEX
            # reduction="sum" so rows weigh by the positions they actually
            # carry; a mean of per-batch means would reweight them whenever
            # the ambiguity mask removes a different number per batch.
            total_loss += float(F.cross_entropy(
                logits.reshape(-1, logits.size(-1))[keep], flat_y[keep],
                reduction="sum"))
            n_scored += int(keep.sum())
            n_ambiguous += int((~keep).sum())
            counts.update(flat_y[keep].tolist())
            if lo % (batch_size * 50) == 0:
                print(f"    rows {lo:>7,} / {len(rows):,}", flush=True)
    return total_loss / n_scored, counts, n_scored, n_ambiguous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--rows", type=int, default=None,
                    help="rows to score; default the whole split")
    ap.add_argument("--modality", default="genome_sequence",
                    help="selects the ambiguous-token list excluded from loss")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, best_at_selection = load_adopted(args.run_dir, device)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ambiguous_ids = resolve_ambiguous_token_ids(
        tok, ambiguous_tokens_for_modality(args.modality))

    rows = load_from_disk(args.dataset_dir)[args.split]
    if args.rows:
        rows = rows.select(range(min(args.rows, len(rows))))

    print(f"  run        {args.run_dir}", flush=True)
    print(f"  checkpoint ckpt.pt  (iter {ck['iter_num']:,}, selected at "
          f"best_val_loss {best_at_selection:.4f})", flush=True)
    print(f"  rows       {len(rows):,} of {args.split}", flush=True)

    loss, counts, n_scored, n_ambiguous = score(
        model, rows, ambiguous_ids, args.batch_size, device)
    baseline = _entropy(counts)
    gc_ids = [tok.convert_tokens_to_ids(b) for b in ("G", "C")]
    gc = sum(counts[i] for i in gc_ids) / n_scored if n_scored else float("nan")

    print(f"  scored {n_scored:,} positions  ambiguous excluded {n_ambiguous:,}"
          f"  GC {100 * gc:.2f}%", flush=True)
    print(f"  → val_loss {loss:.4f}  baseline {baseline:.4f}"
          f"  margin {baseline - loss:+.4f}", flush=True)

    summary = {
        "run_dir": args.run_dir,
        "checkpoint": "ckpt.pt",
        "checkpoint_step": int(ck["iter_num"]),
        "best_val_at_selection": best_at_selection,
        "split": args.split,
        "rows_scored": len(rows),
        "scored_positions": n_scored,
        "ambiguous_excluded": n_ambiguous,
        # The run evaluated under bf16 autocast over 64,000 rows drawn with
        # replacement, and kept the minimum; this pass is fp32 over every row
        # once. The two are not interchangeable and must not be quoted side by
        # side as one number. Every subset is scored by this script, so the
        # comparison across subsets is unaffected.
        "precision": "fp32",
        "selection_precision": "bf16 autocast (as trained)",
        "selection_range": "64,000 rows drawn with replacement, minimum kept",
        "val_loss": loss,
        "degenerate_baseline": baseline,
        "margin": baseline - loss,
        "gc_fraction": gc,
        "label_composition": {tok.convert_ids_to_tokens(k): v / n_scored
                              for k, v in sorted(counts.items())},
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
