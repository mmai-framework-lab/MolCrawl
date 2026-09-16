"""Does raising dataloader_num_workers change which positions get masked?

Turning on four dataloader workers is a placement change -- the same rows in the
same order, moved off the training process. Masking is not placement: the collator
draws it from a random number generator, and with workers the collate runs in the
worker process, whose torch seed PyTorch sets to base_seed + worker_id. If that
makes a different set of positions masked, then a run with four workers differs
from a run with none in more than speed, and the two are not the same condition.

The sampler is sequential on purpose, so both settings see identical rows in an
identical order and the only thing that can differ is the masking draw.

    python scripts/check_mlm_mask_worker_parity.py --config <rna bert config> \
        [--rows 256] [--batch 8] [--workers 4]
"""

from __future__ import annotations

import argparse
import runpy


def masked_positions(dataset, collator, batch: int, workers: int, seed: int) -> list:
    import torch
    from torch.utils.data import DataLoader, SequentialSampler

    from transformers import set_seed

    set_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        num_workers=workers,
        # persistent_workers would keep one generator alive across epochs; a single
        # pass is enough here and leaving it off keeps the two runs comparable.
    )
    out = []
    for b in loader:
        # labels are -100 where the position was left alone, so the masked set is
        # exactly where they are not.
        out.append((b["labels"] != -100).nonzero(as_tuple=False).tolist())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None, help="default: the config's")
    args = ap.parse_args()

    import torch  # noqa: F401  (imported for the side effect of a stable RNG)
    from datasets import load_from_disk
    from transformers import DataCollatorForLanguageModeling

    g = runpy.run_path(args.config, run_name="__main__")
    tokenizer = g["tokenizer"]
    seed = args.seed if args.seed is not None else int(g.get("seed", 42))
    prob = float(g.get("mlm_probability", 0.2))

    ds = load_from_disk(g["dataset_dir"])["train"].select(range(args.rows))
    ds = ds.with_format("torch")
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=prob
    )

    print(f"config      {args.config}")
    print(f"rows {args.rows}  batch {args.batch}  seed {seed}  mlm_probability {prob}")
    print(f"batches     {args.rows // args.batch}\n")

    a = masked_positions(ds, collator, args.batch, 0, seed)
    b = masked_positions(ds, collator, args.batch, args.workers, seed)

    same = a == b
    n_a = sum(len(x) for x in a)
    n_b = sum(len(x) for x in b)
    tokens = args.rows * len(ds[0]["input_ids"])

    print(f"workers=0   masked {n_a:>9,} of {tokens:,} tokens  ({n_a / tokens * 100:.2f} %)")
    print(f"workers={args.workers}   masked {n_b:>9,} of {tokens:,} tokens  ({n_b / tokens * 100:.2f} %)")
    print(f"\nidentical masked positions: {'YES' if same else 'NO'}")

    if not same:
        first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        sa, sb = {tuple(p) for p in a[first]}, {tuple(p) for p in b[first]}
        print(f"  first batch that differs : {first}")
        print(f"  masked in both           : {len(sa & sb):,}")
        print(f"  only without workers     : {len(sa - sb):,}")
        print(f"  only with workers        : {len(sb - sa):,}")
        print("\n  The rate is unchanged, so the objective is the same and the two")
        print("  are statistically equivalent; they are not the same draw, so a run")
        print("  with workers is not a bitwise continuation of one without.")

    # Repeating one setting twice says whether the difference is workers or just
    # non-determinism, which would make the comparison above meaningless.
    b2 = masked_positions(ds, collator, args.batch, args.workers, seed)
    print(f"\nworkers={args.workers} reproduces itself on a rerun: "
          f"{'YES' if b == b2 else 'NO'}")
    a2 = masked_positions(ds, collator, args.batch, 0, seed)
    print(f"workers=0 reproduces itself on a rerun: {'YES' if a == a2 else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
