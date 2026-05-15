#!/usr/bin/env python3
"""Convert an HF-format ``checkpoint-*`` directory to the in-house
``ckpt.pt`` format expected by ``molcrawl.tasks.evaluation._adapters.gpt2``.

The riken GPT class and HF's GPT2LMHeadModel share the same parameter
layout (the HF export was produced from the same minGPT-style code),
so the conversion is a one-step::

    ckpt.pt  =  {
        "model":      torch.load(checkpoint-N/pytorch_model.bin),
        "model_args": json.load(checkpoint-N/config.json)["_riken_model_args"],
    }

Usage::

    python workflows/data/convert-hf-gpt2-to-ckpt.py \\
        <hf_checkpoint_dir> [<output_ckpt_pt>]

When the output path is omitted, ``ckpt.pt`` is written to the
parent directory of the HF checkpoint (the standard
``<modality>/gpt2-output/<size>/ckpt.pt`` location).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert(hf_dir: Path, out_path: Path | None = None) -> Path:
    import torch

    hf_dir = Path(hf_dir).resolve()
    if not hf_dir.is_dir():
        raise SystemExit(f"not a directory: {hf_dir}")
    sd_path = hf_dir / "pytorch_model.bin"
    cfg_path = hf_dir / "config.json"
    if not sd_path.exists():
        raise SystemExit(f"missing pytorch_model.bin in {hf_dir}")
    if not cfg_path.exists():
        raise SystemExit(f"missing config.json in {hf_dir}")

    sd = torch.load(sd_path, map_location="cpu", weights_only=False)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # HF GPT2 stores attention / mlp projection weights as Conv1D
    # (in_features, out_features). The riken minGPT class uses nn.Linear
    # (out_features, in_features). Transpose them so the riken model
    # loads the state_dict cleanly.
    conv1d_suffixes = (
        ".attn.c_attn.weight",
        ".attn.c_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight",
    )
    transposed = 0
    for key in list(sd.keys()):
        if any(key.endswith(s) for s in conv1d_suffixes):
            sd[key] = sd[key].t().contiguous()
            transposed += 1
    if transposed:
        print(f"  transposed {transposed} Conv1D weight tensor(s) to Linear layout")

    model_args = cfg.get("_riken_model_args")
    if not isinstance(model_args, dict):
        # Reconstruct from HF GPT2Config field names.
        model_args = {
            "n_layer": cfg["n_layer"],
            "n_head": cfg["n_head"],
            "n_embd": cfg["n_embd"],
            "block_size": cfg["n_positions"],
            "bias": False,
            "vocab_size": cfg["vocab_size"],
        }
    # Ensure dropout is present (riken GPTConfig expects it; some older
    # HF exports omit it).
    model_args.setdefault("dropout", float(cfg.get("resid_pdrop", 0.1)))

    ckpt = {"model": sd, "model_args": model_args}
    if out_path is None:
        out_path = hf_dir.parent / "ckpt.pt"
    out_path = Path(out_path)
    if out_path.exists():
        print(f"refusing to overwrite existing {out_path}", file=sys.stderr)
        raise SystemExit(1)
    torch.save(ckpt, out_path)
    print(f"wrote {out_path} ({sum(p.numel() for p in sd.values() if hasattr(p, 'numel')):,} params)")
    print(f"  model_args: {model_args}")
    return out_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("hf_dir", help="HF checkpoint directory (must contain pytorch_model.bin + config.json)")
    p.add_argument("out", nargs="?", default=None, help="Output ckpt.pt path (default: parent_dir/ckpt.pt)")
    args = p.parse_args(argv)
    convert(Path(args.hf_dir), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
