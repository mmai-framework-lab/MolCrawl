"""Recompute the MLM judgement baselines, from the data alone.

The compounds BERT work is judged against four yardsticks at `[MASK]` positions -
unigram, left-neighbour look-up, both-neighbour look-up, and the degenerate blended
loss. They were derived ad hoc in August and no script survived, so what slice they
were measured on, and whether the neighbours used were the corrupted or the original
tokens, could not be checked. Both matter now that the eval slice is drawn at random
and the endpoints have been re-measured.

Nothing here needs a model: every baseline is a count table fitted on train and scored
on the eval slice, at exactly the positions the collator masks.

Two readings of "look-up" are reported, because the earlier derivation does not say
which it used:

* ``observed``  - condition on the neighbours the model actually sees, which may
                  themselves be masked or randomly replaced. This is what a real
                  predictor has to work with.
* ``oracle``    - condition on the original tokens. An upper bound on look-up.

Usage:
    LEARNING_SOURCE_DIR=<dir> python scripts/mlm_lookup_baselines.py \
        --dataset-dir <training_ready_bert> --label packed
"""
from __future__ import annotations

import json
import logging
import math
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlm_baselines")

SMOOTHING = 1e-4  # backoff weight on the unigram, so an unseen context is not -inf


def fit_tables(rows, vocab_size, special_ids, unigram_only=False):
    """Unigram, left-conditional and both-conditional counts over the training rows.

    With ``unigram_only`` the two conditional tables are left empty: the callers
    that only need the unigram entropy and the degenerate threshold then run at a
    fit size the context tables could not be held at.
    """
    uni = np.zeros(vocab_size, dtype=np.float64)
    left = defaultdict(Counter)
    both = defaultdict(Counter)
    for ids in rows:
        ids = list(ids)
        for i, t in enumerate(ids):
            if t in special_ids:
                continue
            uni[t] += 1
            if unigram_only:
                continue
            if i:
                left[ids[i - 1]][t] += 1
            if i and i + 1 < len(ids):
                both[(ids[i - 1], ids[i + 1])][t] += 1
    uni /= max(uni.sum(), 1)
    return uni, left, both


def tokenizer_from_config(config_path):
    """Run a pretrain config and hand back the tokenizer it builds.

    The configs are executed, not parsed: each one builds its tokenizer in a
    different way (rna wraps a WordLevel gene dictionary, molnl loads a
    pretrained one, compounds reads a vocab file), and the only thing that is
    the same across them is the name the trainer reads it under. Running the
    file is what the trainer does too, so the baseline cannot drift from the
    run by picking the tokenizer up a second way.

    Returns the tokenizer and the config's namespace, so dataset_dir,
    max_length and mlm_probability come from the same place.
    """
    import runpy

    ns = runpy.run_path(str(config_path))
    tokenizer = ns.get("tokenizer")
    if tokenizer is None:
        raise SystemExit(f"{config_path} defines no `tokenizer`")
    return tokenizer, ns


def _score(counter, token, uni):
    """-log p(token | context), backing off to the unigram."""
    total = sum(counter.values()) if counter else 0
    p_ctx = counter.get(token, 0) / total if total else 0.0
    p = (1 - SMOOTHING) * p_ctx + SMOOTHING * uni[token]
    return -math.log(max(p, 1e-12))


def main() -> int:
    parser = ArgumentParser(description="MLM look-up baselines at [MASK] positions")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "a pretrain BERT config. Its tokenizer, dataset_dir, max_length and "
            "mlm_probability are taken as the run used them, so the baseline is "
            "fitted and scored exactly where the model was. Without it the "
            "compounds tokenizer and --dataset-dir are used, as before."
        ),
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--fit-rows", type=int, default=200000, help="training rows the tables are fitted on")
    parser.add_argument(
        "--unigram-only",
        action="store_true",
        help=(
            "fit the unigram alone and skip the neighbour tables. The two "
            "conditional tables key on one and two neighbouring token ids, so on a "
            "25k or 50k vocabulary they grow with the number of distinct contexts "
            "seen and not with the vocabulary -- hundreds of millions of keys at "
            "the default fit size. The unigram entropy and the degenerate "
            "threshold do not use them, so this fits those two on many more rows "
            "than the full run can hold."
        ),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--mlm-probability", type=float, default=0.2)
    parser.add_argument("--vocab", default="assets/molecules/vocab.txt")
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not (args.config or args.dataset_dir):
        parser.error("give --config (preferred) or --dataset-dir")

    import torch
    from datasets import load_from_disk

    from molcrawl.models._collators import make_mlm_collator
    # -100, the value HF's MLM collator fills unscored label positions with. Not the
    # IGNORE_INDEX in _collators.ambiguity_aware_collator, which is -1 and belongs to
    # the CLM path: comparing MLM labels against that makes every position look scored,
    # so the neighbour look-ups read -100 as their context and miss ~80% of the time.
    from molcrawl.models.bert._mlm_diagnostics import IGNORE_INDEX
    from molcrawl.models.bert.main import resolve_eval_split, subsample_eval_split

    if args.config:
        tokenizer, cfg = tokenizer_from_config(args.config)
        dataset_dir = args.dataset_dir or cfg.get("dataset_dir")
        if not dataset_dir:
            parser.error(f"{args.config} declares no dataset_dir; pass --dataset-dir")
        # main.py pins mlm_probability_effective to 0.2 for every modality and no
        # config overrides it today, so this reads 0.2 everywhere. Taken from the
        # config anyway, so that a modality which starts declaring its own rate is
        # scored at that rate instead of silently at 0.2.
        mlm_probability = cfg.get("mlm_probability", args.mlm_probability)
        # main.py does not read ambiguous_tokens from the config either: it infers
        # the modality from dataset_dir and looks the list up. Taken the same way
        # here, so the positions this scores are the positions the run scored.
        from molcrawl.models._collators import (
            ambiguous_tokens_for_modality,
            infer_modality_from_path,
        )

        modality = infer_modality_from_path(dataset_dir) or infer_modality_from_path(
            cfg.get("model_path")
        )
        ambiguous = ambiguous_tokens_for_modality(modality) if modality else []
        logger.info("modality %r, ambiguous_tokens %s", modality, ambiguous)
    else:
        from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer

        tokenizer = CompoundsTokenizer(args.vocab, args.max_length)
        dataset_dir = args.dataset_dir
        mlm_probability = args.mlm_probability
        ambiguous = []

    dataset = load_from_disk(dataset_dir)
    split = args.split or resolve_eval_split(dataset.keys())
    eval_rows = subsample_eval_split(dataset[split], random_sample=True)

    actual = getattr(tokenizer, "tokenizer", tokenizer)
    vocab_size = len(actual)
    special_ids = set(actual.all_special_ids)
    logger.info(
        "%s: vocab %d, mlm_probability %g, specials %s",
        args.label or dataset_dir, vocab_size, mlm_probability, sorted(special_ids),
    )

    train = dataset["train"]
    n_fit = min(args.fit_rows, len(train))
    logger.info("fitting count tables on %d train rows", n_fit)
    uni, left, both = fit_tables(
        train[:n_fit]["input_ids"], vocab_size, special_ids, unigram_only=args.unigram_only
    )

    # Mask exactly where the collator would, with the seed the endpoint re-scores used.
    collator = make_mlm_collator(actual, ambiguous_tokens=ambiguous, mlm_probability=mlm_probability)
    torch.manual_seed(0)
    mask_id = actual.mask_token_id

    sums = {k: 0.0 for k in ("unigram", "left_observed", "both_observed", "left_oracle", "both_oracle")}
    count = 0
    blended_sum, blended_count = 0.0, 0
    step = 64
    for start in range(0, len(eval_rows), step):
        chunk = [eval_rows[i] for i in range(start, min(start + step, len(eval_rows)))]
        batch = collator(chunk)
        corrupted = batch["input_ids"].numpy()
        labels = batch["labels"].numpy()
        for r in range(corrupted.shape[0]):
            row_c, row_l = corrupted[r], labels[r]
            original = np.where(row_l != IGNORE_INDEX, row_l, row_c)
            for i in np.flatnonzero(row_l != IGNORE_INDEX):
                truth = int(row_l[i])
                # The blended baseline is the degenerate solution: unigram everywhere
                # except the copy positions, which are free.
                blended_sum += 0.0 if row_c[i] == truth else -math.log(max(uni[truth], 1e-12))
                blended_count += 1
                if row_c[i] != mask_id:
                    continue  # [MASK] positions only, matching the endpoint metric
                count += 1
                sums["unigram"] += -math.log(max(uni[truth], 1e-12))
                # A position with no neighbour to condition on falls back to the
                # unigram for that token - the same thing a look-up table would do.
                unigram_here = -math.log(max(uni[truth], 1e-12))
                for tag, src in (("observed", row_c), ("oracle", original)):
                    lft = int(src[i - 1]) if i else None
                    rgt = int(src[i + 1]) if i + 1 < len(src) else None
                    sums[f"left_{tag}"] += (
                        _score(left.get(lft, Counter()), truth, uni) if lft is not None else unigram_here
                    )
                    ctx = (lft, rgt) if lft is not None and rgt is not None else None
                    sums[f"both_{tag}"] += (
                        _score(both.get(ctx, Counter()), truth, uni) if ctx else unigram_here
                    )
        if start and start % (step * 40) == 0:
            logger.info("    %d/%d rows, unigram %.4f", start, len(eval_rows), sums["unigram"] / count)

    result = {
        "label": args.label,
        "config": args.config,
        "dataset_dir": dataset_dir,
        "vocab_size": vocab_size,
        "mlm_probability": mlm_probability,
        "unigram_only": bool(args.unigram_only),
        "ambiguous_tokens": list(ambiguous),
        "split": split,
        "eval_rows": len(eval_rows),
        "fit_rows": n_fit,
        "masked_positions": count,
        "scored_positions": blended_count,
        "degenerate_blended": blended_sum / blended_count,
        **{k: v / count for k, v in sums.items() if not (args.unigram_only and k != "unigram")},
    }

    print(f"\n=== {args.label or dataset_dir} — [MASK] 位置の基準線 ===")
    print(f"  unigram              {result['unigram']:.4f}")
    if args.unigram_only:
        # The conditional tables were never fitted, so the look-up columns would
        # be the unigram seen through the backoff weight -- not a baseline.
        print("  left/both-neighbour  （--unigram-only のため未測定）")
    else:
        print(f"  left-neighbour  観測 {result['left_observed']:.4f}   oracle {result['left_oracle']:.4f}")
        print(f"  both-neighbour  観測 {result['both_observed']:.4f}   oracle {result['both_oracle']:.4f}")
    print(f"  degenerate (混合 loss) {result['degenerate_blended']:.4f}")
    print(f"  非コピー比 {result['degenerate_blended'] / result['unigram']:.4f}")
    print(f"  masked positions {count:,}   scored positions {blended_count:,}   eval rows {len(eval_rows):,}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
