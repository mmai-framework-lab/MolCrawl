#!/usr/bin/env python3
"""Where the molnl retrieval scoring got its tokenizers, checkpoints and pairs.

The 2026-11-09 judgment asks four questions that the scoring runs did not record, and
says no new scoring or scan is needed for any of them. Each is answered by loading the
same thing the scoring loaded and printing what came back, rather than by reading the
code and asserting what it would do.

  pairs      the pair-level overlap was judged on the description as written and the
             molecule after canonicalising both sides, so the 83 molecules RDKit will
             not read could not have matched. Whether that leaves a gap depends on
             whether any of those 83 sit at a pair position whose description was found
             in the corpus -- 18 positions of the 1,000. If none do, the answer is 0 of
             1,000 with nothing left undecided.
  tokenizer  the adapters resolve a tokenizer in tiers and log which tier answered at
             INFO, which the scoring jobs did not capture. Each tier is tried here with
             logging on, and the tokenizer that answers is compared against the one the
             pretraining used: vocabulary size, then the ids of probe strings.
  steps      a nanoGPT run keeps one ckpt.pt whose step is inside the file. Read with
             mmap so an xl checkpoint is not pulled through memory to get one integer.
  inventory  which checkpoints across the modalities carry a tokenizer of their own and
             which fall back to something resolved at scoring time. Taken by walking the
             output root rather than from a list of the ones we happen to remember.

Writes one JSON and prints the same content. No GPU, no model weights beyond the
nanoGPT metadata read.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

PROBES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "The molecule is a benzoate ester.",
    "<s>[INST] CCO [/INST]ethanol<|endoftext|>",
]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt",
                   "merges.txt", "special_tokens_map.json", "spiece.model")


def probe_ids(tokenizer):
    """The ids a tokenizer gives the probe strings, without added specials."""
    out = {}
    for text in PROBES:
        try:
            out[text] = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception as exc:  # noqa: BLE001
            try:
                out[text] = list(tokenizer.encode(text))
            except Exception as inner:  # noqa: BLE001
                out[text] = f"encode failed: {exc} / {inner}"
    return out


def describe(tokenizer, where):
    return {
        "resolved_from": where,
        "class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "len": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "probe_ids": probe_ids(tokenizer),
    }


def pair_question(pairs_file, index_file):
    """Does any RDKit-unreadable molecule sit where the description was found?"""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    pairs = [json.loads(line) for line in open(pairs_file, encoding="utf-8")
             if line.strip()]
    unreadable = [i for i, p in enumerate(pairs)
                  if Chem.MolFromSmiles(p["molecule"]) is None]
    index = json.load(open(index_file, encoding="utf-8"))
    found = index.get("description_in_corpus", [])
    both = sorted(set(unreadable) & set(found))
    return {
        "pairs": len(pairs),
        "molecules_rdkit_would_not_read": len(unreadable),
        "positions_whose_description_is_in_the_corpus": len(found),
        "positions_needing_both": both,
        "pair_question_decided_for": len(pairs) - len(both),
        "reading": ("every position whose description is absent cannot be a pair match "
                    "whatever its molecule, so only the overlap of the two lists could "
                    "leave the pair question open"),
    }


def scoring_tokenizers(modality):
    """The tokenizer each arch's adapter hands the scorer, and the pretraining one."""
    from molcrawl.tasks.evaluation import _adapters  # noqa: F401
    from molcrawl.tasks.evaluation._base import ModelHandle
    from molcrawl.tasks.evaluation._base.model_adapter import build_adapter

    out = {}
    for arch, model_path in (("bert", os.environ.get("BERT_CKPT")),
                             ("gpt2", os.environ.get("GPT2_CKPT"))):
        if not model_path:
            out[arch] = {"skipped": f"set {arch.upper()}_CKPT"}
            continue
        handle = ModelHandle(arch=arch, modality=modality, model_path=model_path,
                             tokenizer_path=None, size=None,
                             extras={"device": "cpu"})
        adapter = build_adapter(handle)
        # The two adapters name the step differently -- the MLM one walks tiers in
        # _load_tokenizer, the GPT-2 one builds per modality in _build_tokenizer -- and
        # assuming one name left the GPT-2 side of this question unanswered once.
        try:
            if hasattr(adapter, "_load_tokenizer"):
                tokenizer = adapter._load_tokenizer()  # noqa: SLF001
                where = f"_load_tokenizer tiers for ({arch}, {modality})"
            elif hasattr(adapter, "_build_tokenizer"):
                tokenizer = adapter._build_tokenizer(modality)[0]  # noqa: SLF001
                where = f"_build_tokenizer({modality}) for {arch}"
            else:
                raise AttributeError(
                    f"{type(adapter).__name__} exposes neither _load_tokenizer nor "
                    "_build_tokenizer")
            out[arch] = describe(tokenizer, where)
        except Exception as exc:  # noqa: BLE001
            out[arch] = {"error": f"{type(exc).__name__}: {exc}"}
        out[arch]["checkpoint"] = model_path
        out[arch]["checkpoint_carries_tokenizer"] = sorted(
            f for f in TOKENIZER_FILES
            if os.path.exists(os.path.join(
                model_path if os.path.isdir(model_path) else os.path.dirname(model_path),
                f)))
    from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

    pre = MoleculeNatLangTokenizer()
    inner = getattr(pre, "tokenizer", pre)
    out["pretraining"] = describe(inner, "MoleculeNatLangTokenizer(), as the configs use")
    out["pretraining"]["GPT2_TOKENIZER_DIR"] = os.environ.get("GPT2_TOKENIZER_DIR")
    for arch in ("bert", "gpt2"):
        side = out.get(arch, {})
        if "probe_ids" in side:
            out[arch]["same_ids_as_pretraining"] = (
                side["probe_ids"] == out["pretraining"]["probe_ids"])
            out[arch]["same_vocab_size_as_pretraining"] = (
                side["vocab_size"] == out["pretraining"]["vocab_size"])
    return out


def nanogpt_steps(paths):
    """iter_num and the saved val loss out of each ckpt.pt, without loading weights."""
    import torch

    out = {}
    for path in paths:
        if not path or not os.path.exists(path):
            out[path] = {"error": "not on disk"}
            continue
        try:
            try:
                blob = torch.load(path, map_location="cpu", mmap=True,
                                  weights_only=False)
            except TypeError:  # torch without mmap=
                blob = torch.load(path, map_location="cpu")
            out[path] = {k: (v if isinstance(v, (int, float, str)) else str(type(v)))
                         for k, v in blob.items() if k != "model"}
            out[path]["iter_num"] = blob.get("iter_num")
            best = blob.get("best_val_loss")
            out[path]["best_val_loss"] = float(best) if best is not None else None
            del blob
        except Exception as exc:  # noqa: BLE001
            out[path] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def tokenizer_inventory(root, limit=4000):
    """Every checkpoint under root, and whether a tokenizer sits with it."""
    hf, nano = [], []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "config.json" in files and any(
                f.startswith("model") and f.endswith((".safetensors", ".bin"))
                for f in files):
            hf.append({"dir": base,
                       "tokenizer_files": sorted(set(files) & set(TOKENIZER_FILES))})
        if "ckpt.pt" in files:
            nano.append({"dir": base,
                         "tokenizer_files": sorted(set(files) & set(TOKENIZER_FILES))})
        if len(hf) + len(nano) > limit:
            break
    def split(rows):
        with_tok = [r for r in rows if r["tokenizer_files"]]
        without = [r for r in rows if not r["tokenizer_files"]]
        return {"total": len(rows), "with_tokenizer": len(with_tok),
                "without_tokenizer": len(without),
                "without_tokenizer_examples": [r["dir"] for r in without[:10]],
                "with_tokenizer_examples": [r["dir"] for r in with_tok[:10]]}
    return {"root": root, "hf_checkpoints": split(hf), "nanogpt_checkpoints": split(nano)}


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="  [%(name)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file")
    ap.add_argument("--pair-index")
    ap.add_argument("--modality", default="molecule_nat_lang")
    ap.add_argument("--gpt2-ckpt", action="append", default=[],
                    help="a nanoGPT ckpt.pt to read the step out of; repeatable")
    ap.add_argument("--inventory-root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    answers = {}
    if a.pairs_file and a.pair_index:
        print("== pairs ==", flush=True)
        answers["pair_question"] = pair_question(a.pairs_file, a.pair_index)
        print(json.dumps(answers["pair_question"], ensure_ascii=False, indent=2),
              flush=True)
    print("== tokenizers ==", flush=True)
    answers["tokenizers"] = scoring_tokenizers(a.modality)
    print(json.dumps(answers["tokenizers"], ensure_ascii=False, indent=2), flush=True)
    if a.gpt2_ckpt:
        print("== nanoGPT steps ==", flush=True)
        answers["nanogpt_steps"] = nanogpt_steps(a.gpt2_ckpt)
        print(json.dumps(answers["nanogpt_steps"], ensure_ascii=False, indent=2,
                         default=str), flush=True)
    if a.inventory_root:
        print("== tokenizer inventory ==", flush=True)
        answers["tokenizer_inventory"] = tokenizer_inventory(a.inventory_root)
        print(json.dumps(answers["tokenizer_inventory"], ensure_ascii=False, indent=2),
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(answers, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
