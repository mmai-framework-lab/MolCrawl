"""Phase 0-5 audit: resolve each production config and show a resolved-value matrix.

Runs each (modality × arch × size) config module and extracts the training
knobs relevant to the production spec. Prints a matrix + expected values.

Modality scope (per production spec): rna, genome_sequence (Evo2 subset),
protein_sequence, molecule_nat_lang, compounds/organix13.

For each config we report resolved values of:
  - learning_rate
  - max_iters (GPT-2) / max_steps (BERT)
  - warmup_iters / warmup_steps
  - min_lr
  - beta1, beta2 (or adam_beta1/2)
  - grad_clip
  - weight_decay
  - block_size (GPT-2) / max_length (BERT)
  - meta_vocab_size (rna GPT-2 only — expected 25,426)

Fatal errors during import are caught and reported.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
sys.path.insert(0, str(REPO))
os.environ.setdefault("LEARNING_SOURCE_DIR", "/lustre/home/matsubara/learning_source_20260520_uncapped")

CONFIGS = [
    # (modality, arch, size, module_path)
    ("rna",               "gpt2", "small",  "molcrawl.tasks.pretrain.configs.rna.gpt2_small"),
    ("rna",               "gpt2", "medium", "molcrawl.tasks.pretrain.configs.rna.gpt2_medium"),
    ("rna",               "gpt2", "large",  "molcrawl.tasks.pretrain.configs.rna.gpt2_large"),
    ("rna",               "gpt2", "xl",     "molcrawl.tasks.pretrain.configs.rna.gpt2_xl"),
    ("rna",               "bert", "small",  "molcrawl.tasks.pretrain.configs.rna.bert_small"),
    ("rna",               "bert", "medium", "molcrawl.tasks.pretrain.configs.rna.bert_medium"),
    ("rna",               "bert", "large",  "molcrawl.tasks.pretrain.configs.rna.bert_large"),
    ("rna",               "bert", "xl",     "molcrawl.tasks.pretrain.configs.rna.bert_xl"),
    ("genome_sequence",   "bert", "small_subset", "molcrawl.tasks.pretrain.configs.genome_sequence.bert_small_subset"),
    ("genome_sequence",   "bert", "xl_subset",    "molcrawl.tasks.pretrain.configs.genome_sequence.bert_xl_subset"),
    ("genome_sequence",   "gpt2", "small_subset", "molcrawl.tasks.pretrain.configs.genome_sequence.gpt2_small_subset"),
    ("protein_sequence",  "gpt2", "small",  "molcrawl.tasks.pretrain.configs.protein_sequence.gpt2_small"),
    ("protein_sequence",  "gpt2", "medium", "molcrawl.tasks.pretrain.configs.protein_sequence.gpt2_medium"),
    ("protein_sequence",  "gpt2", "large",  "molcrawl.tasks.pretrain.configs.protein_sequence.gpt2_large"),
    ("protein_sequence",  "gpt2", "xl",     "molcrawl.tasks.pretrain.configs.protein_sequence.gpt2_xl"),
    ("protein_sequence",  "bert", "small",  "molcrawl.tasks.pretrain.configs.protein_sequence.bert_small"),
    ("protein_sequence",  "bert", "medium", "molcrawl.tasks.pretrain.configs.protein_sequence.bert_medium"),
    ("protein_sequence",  "bert", "large",  "molcrawl.tasks.pretrain.configs.protein_sequence.bert_large"),
    ("protein_sequence",  "bert", "xl",     "molcrawl.tasks.pretrain.configs.protein_sequence.bert_xl"),
    ("molecule_nat_lang", "gpt2", "small",  "molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_small"),
    ("molecule_nat_lang", "gpt2", "medium", "molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_medium"),
    ("molecule_nat_lang", "gpt2", "large",  "molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_large"),
    ("molecule_nat_lang", "gpt2", "xl",     "molcrawl.tasks.pretrain.configs.molecule_nat_lang.gpt2_xl"),
    ("molecule_nat_lang", "bert", "small",  "molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_small"),
    ("molecule_nat_lang", "bert", "medium", "molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_medium"),
    ("molecule_nat_lang", "bert", "large",  "molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_large"),
    ("molecule_nat_lang", "bert", "xl",     "molcrawl.tasks.pretrain.configs.molecule_nat_lang.bert_xl"),
    ("compounds",         "gpt2", "small",  "molcrawl.tasks.pretrain.configs.compounds.gpt2_small"),
    ("compounds",         "gpt2", "medium", "molcrawl.tasks.pretrain.configs.compounds.gpt2_medium"),
    ("compounds",         "gpt2", "large",  "molcrawl.tasks.pretrain.configs.compounds.gpt2_large"),
    ("compounds",         "gpt2", "xl",     "molcrawl.tasks.pretrain.configs.compounds.gpt2_xl"),
    ("compounds",         "bert", "small",  "molcrawl.tasks.pretrain.configs.compounds.bert_small"),
    ("compounds",         "bert", "medium", "molcrawl.tasks.pretrain.configs.compounds.bert_medium"),
    ("compounds",         "bert", "large",  "molcrawl.tasks.pretrain.configs.compounds.bert_large"),
    ("compounds",         "bert", "xl",     "molcrawl.tasks.pretrain.configs.compounds.bert_xl"),
]

# Fields we care about, in display order.
FIELDS = [
    "learning_rate", "max_iters", "max_steps",
    "warmup_iters", "warmup_steps",
    "min_lr", "beta1", "beta2", "grad_clip",
    "weight_decay",
    "block_size", "max_length",
    "meta_vocab_size",
]


def resolve(module_path: str) -> dict:
    """Import a config module and return module-level scalar globals."""
    # Fresh import each time to avoid globals sharing across sizes.
    if module_path in sys.modules:
        del sys.modules[module_path]
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}
    out = {}
    for f in FIELDS:
        if hasattr(mod, f):
            v = getattr(mod, f)
            if isinstance(v, (int, float, str, bool)):
                out[f] = v
    return out


def main() -> int:
    print("=== Phase 0-5 audit: resolved values per config ===\n")
    # header
    hdr = ["modality", "arch", "size"] + FIELDS
    widths = [15, 5, 14] + [12] * len(FIELDS)
    print(" | ".join(h.ljust(w) for h, w in zip(hdr, widths)))
    print("-+-".join("-" * w for w in widths))

    all_rows = []
    for modality, arch, size, path in CONFIGS:
        vals = resolve(path)
        if "__error__" in vals:
            row = [modality, arch, size] + ["ERR"] * len(FIELDS)
            print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))
            print(f"    error: {vals['__error__']}")
            continue
        row = [modality, arch, size] + [
            (f"{vals[f]:.4g}" if isinstance(vals.get(f), float)
             else str(vals.get(f, "-"))) for f in FIELDS
        ]
        print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))
        all_rows.append((modality, arch, size, vals))

    # rna GPT-2 vocab check
    print("\n=== rna GPT-2 vocab check ===")
    for modality, arch, size, vals in all_rows:
        if modality == "rna" and arch == "gpt2":
            mvs = vals.get("meta_vocab_size", "-")
            ok = "✅" if mvs == 25426 else "⚠"
            print(f"  {modality}/{arch}/{size}: meta_vocab_size = {mvs}  {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
