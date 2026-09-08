"""Final scoring for a finished genome BERT run.

Selection and scoring answer different questions and do not need the same range.
A run picks its checkpoint from the first 10,000 valid rows, which is what it
evaluated on throughout; the number the campaign then compares across subsets is
measured on that subset's whole valid split. The two ranges are not
interchangeable -- on mammal_centered they differ by 0.0023 against a run-to-run
noise of 0.0011, because the first 10,000 rows carry 41.09% GC and all 50,000
carry 40.46%.

``eval_loss_mask`` is a mean over the positions a random draw happened to mask,
so one draw is one sample. Three seeds are scored and both the mean and the
spread are reported: the spread says whether a difference between two subsets is
larger than the masking draw itself, which a single number cannot.

The degenerate baseline is computed in the same pass, from the labels at exactly
the positions that were masked. Saving the positions and recomputing later would
be the same arithmetic with an opportunity for the two to drift apart; here the
baseline cannot be measured anywhere other than where the model was measured.
"""
import argparse
import json
import math
import os
from collections import Counter

import torch

from molcrawl.models.bert._mlm_diagnostics import IGNORE_INDEX, split_mlm_loss


def _entropy(counts):
    """Cross-entropy paid by predicting the marginal of these very labels."""
    total = sum(counts.values())
    if not total:
        return None
    return -sum((c / total) * math.log(c / total) for c in counts.values() if c)


def _best_checkpoint(run_dir):
    """The checkpoint the run itself selected, on the range it evaluated on."""
    if not os.path.isdir(run_dir):
        raise SystemExit(f"{run_dir}: no such run directory")
    dirs = [d for d in os.listdir(run_dir)
            if d.startswith("checkpoint-") and d.split("-")[1].isdigit()]
    if not dirs:
        raise SystemExit(f"{run_dir}: holds no checkpoint-* directory to score")
    latest = max(dirs, key=lambda d: int(d.split("-")[1]))
    state = json.load(open(os.path.join(run_dir, latest, "trainer_state.json")))
    best = state.get("best_model_checkpoint")
    if best and os.path.isdir(best):
        return best, state
    if best:                                  # moved tree: same basename, new root
        cand = os.path.join(run_dir, os.path.basename(best))
        if os.path.isdir(cand):
            return cand, state
    raise SystemExit(f"{run_dir}: best_model_checkpoint {best!r} is not on disk")


def score(model, collator, rows, seed, batch_size, device, mask_token_id):
    """One masking draw over every row, scored the way training scores it.

    The loss itself comes from ``split_mlm_loss`` -- the same function the Trainer
    mixin uses during the run -- so the number here cannot drift from the one the
    checkpoint was selected on. Only the label tally at the masked positions is
    added, and that is what the degenerate baseline is computed from.
    """
    torch.manual_seed(seed)
    sums = Counter()
    counts = Counter()
    label_counts = Counter()
    with torch.no_grad():
        for lo in range(0, len(rows), batch_size):
            batch = [{"input_ids": r} for r in rows[lo:lo + batch_size]["input_ids"]]
            enc = collator(batch)
            ids = enc["input_ids"].to(device)
            labels = enc["labels"].to(device)
            attn = enc.get("attention_mask")
            out = model(input_ids=ids,
                        attention_mask=attn.to(device) if attn is not None else None)

            batch_sums, batch_counts = split_mlm_loss(
                out.logits, labels, ids, mask_token_id)
            for k in ("mask", "copy", "random"):
                sums[k] += batch_sums[k]
                counts[k] += batch_counts[k]

            scored = labels != IGNORE_INDEX
            if bool(scored.any()):
                flat_labels = labels[scored]
                is_mask = ids[scored] == mask_token_id
                if bool(is_mask.any()):
                    label_counts.update(flat_labels[is_mask].tolist())

    if not counts["mask"]:
        raise SystemExit("no [MASK] positions were produced; check mlm_probability")
    return {
        "seed": seed,
        "masked_positions": counts["mask"],
        "eval_loss_mask": sums["mask"] / counts["mask"],
        "eval_loss_copy": (sums["copy"] / counts["copy"]) if counts["copy"] else None,
        "eval_loss_random": (sums["random"] / counts["random"]) if counts["random"] else None,
        "degenerate_baseline": _entropy(label_counts),
        "label_counts": {int(k): int(v) for k, v in sorted(label_counts.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--rows", type=int, default=None,
                    help="rows to score; default the whole split")
    ap.add_argument("--seeds", default="1026,1027,1028")
    ap.add_argument("--mlm-probability", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--checkpoint", default="",
                    help="override; default is the run's own best_model_checkpoint")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer, BertForMaskedLM

    from molcrawl.models._collators import (ambiguous_tokens_for_modality,
                                            make_mlm_collator)

    ckpt, state = (args.checkpoint, None) if args.checkpoint else _best_checkpoint(args.run_dir)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    collator = make_mlm_collator(
        tok,
        ambiguous_tokens=ambiguous_tokens_for_modality("genome_sequence"),
        mlm_probability=args.mlm_probability,
    )
    rows = load_from_disk(args.dataset_dir)[args.split]
    if args.rows:
        rows = rows.select(range(min(args.rows, len(rows))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForMaskedLM.from_pretrained(ckpt).to(device).eval()

    print(f"  run        {args.run_dir}")
    sel = None if state is None else state.get("best_metric")
    print(f"  checkpoint {os.path.basename(ckpt)}"
          + ("" if sel is None else f"  (selected at eval_loss_mask {sel:.4f}"
                                    f" on the first-10,000-row range)"), flush=True)
    print(f"  rows       {len(rows):,} of {args.split}", flush=True)
    print(f"  mlm_prob   {args.mlm_probability}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    runs = []
    for s in seeds:
        r = score(model, collator, rows, s, args.batch_size, device, tok.mask_token_id)
        r["margin"] = r["degenerate_baseline"] - r["eval_loss_mask"]
        runs.append(r)
        print(f"  seed {s}  masked {r['masked_positions']:>12,}"
              f"  eval_loss_mask {r['eval_loss_mask']:.4f}"
              f"  baseline {r['degenerate_baseline']:.4f}"
              f"  margin {r['margin']:+.4f}", flush=True)

    def agg(k):
        v = [r[k] for r in runs]
        return {"mean": sum(v) / len(v), "min": min(v), "max": max(v),
                "spread": max(v) - min(v)}

    summary = {
        "run_dir": args.run_dir,
        "checkpoint": ckpt,
        "best_metric_at_selection": None if state is None else state.get("best_metric"),
        "split": args.split,
        "rows_scored": len(rows),
        "mlm_probability": args.mlm_probability,
        # The run evaluated under bf16 autocast; this pass is fp32. Values here
        # are therefore not expected to reproduce best_metric_at_selection even
        # at the same rows and seed. Every subset is scored by this script, so
        # the comparison across subsets is unaffected -- but the two numbers are
        # not interchangeable and should not be quoted side by side as one.
        "precision": "fp32",
        "selection_precision": "bf16 autocast (as trained)",
        "seeds": seeds,
        "per_seed": runs,
        "eval_loss_mask": agg("eval_loss_mask"),
        "degenerate_baseline": agg("degenerate_baseline"),
        "margin": agg("margin"),
    }
    e, m = summary["eval_loss_mask"], summary["margin"]
    print(f"  → eval_loss_mask {e['mean']:.4f}  spread {e['spread']:.4f}"
          f"   margin {m['mean']:+.4f}  spread {m['spread']:.4f}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
