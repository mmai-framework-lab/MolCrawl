"""Build a nanoGPT-format ckpt.pt from a run's MAX-STEP HF checkpoint.

NOT USED. The max-step weights are already available in nanoGPT format at
``checkpoint-<step>/training_state.bin`` (dict with model/model_args/config/iter_num),
so the scorer reads that file directly and this repack is unnecessary. Kept for
reference only. See review-10-bert-arch-inventory-2026-09-14 §3.2 and
protein-arch-inventory-verdict-2026-09-15 §2.


Why this exists
---------------
The protein GPT-2 retrain (21 runs) saved two things per run:

* ``ckpt.pt`` — nanoGPT format, written at *best_val*. For medium/large that is
  NOT the max step (e.g. medium best_val=30,150 while the run reached 33,500).
* ``checkpoint-<step>/`` — HF format (config.json + pytorch_model.bin), saved
  periodically, so the MAX-STEP weights live only here.

The ladder scorer ``eval_gpt2_perplexity.py`` reads the nanoGPT ``ckpt.pt``
layout (``model_args`` / ``model`` / ``config``). To score the MAX STEP that the
boss's ladder point asks for, we take the run's own ``ckpt.pt`` (which carries the
correct ``model_args`` and the ``config.dataset`` string the scorer needs to
exclude X/B/Z from the loss), and swap in the max-step HF weights. Only the
weights change; every other field the scorer relies on is preserved.

HF stores the attention/MLP projection matrices as Conv1D (in, out); nanoGPT uses
Linear (out, in). Rather than hard-code which keys transpose, each HF weight is
matched to the base state_dict's shape: transpose only when the transpose is what
fits, and refuse if neither orientation matches (so a silent misload is caught).

Output is a ckpt.pt this repo's scorer accepts verbatim via --checkpoint-template.
No GPU needed; runs on a compute node (never the login node).
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch


def find_max_hf_checkpoint(run_dir: Path) -> Path:
    dirs = glob.glob(str(run_dir / "checkpoint-*"))
    steps = []
    for d in dirs:
        base = os.path.basename(d)
        try:
            steps.append((int(base.split("-")[1]), Path(d)))
        except (IndexError, ValueError):
            continue
    if not steps:
        raise SystemExit(f"no checkpoint-<step>/ dirs under {run_dir}")
    steps.sort(key=lambda x: x[0])
    return steps[-1][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="a retrain2-gpt2-* run dir (has ckpt.pt + checkpoint-<step>/)")
    ap.add_argument("--out", required=True, help="where to write the nanoGPT-format ckpt.pt for scoring")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    base_ckpt_path = run_dir / "ckpt.pt"
    if not base_ckpt_path.exists():
        raise SystemExit(f"no ckpt.pt in {run_dir}; need it for model_args/config")

    hf_dir = find_max_hf_checkpoint(run_dir)
    max_step = int(hf_dir.name.split("-")[1])

    base = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    hf_sd = torch.load(hf_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)

    # HF stores these four as Conv1D (in, out); nanoGPT uses Linear (out, in), so the
    # weight is the transpose. This is the nanoGPT from_pretrained convention, applied
    # by NAME rather than by shape: attn.c_proj is square (n_embd, n_embd), so a
    # shape-based guess cannot tell it needs transposing and would load it wrong
    # (that produced valid losses 0.8-41 above best_val before this fix). Note wpe is
    # also square but is an embedding and must NOT transpose, which is why the list is
    # explicit rather than "transpose every square Conv1D-looking matrix".
    TRANSPOSE = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

    base_model = base["model"]  # nanoGPT state_dict, keys may carry a wrapper prefix
    new_model = {}
    used = set()
    for key, base_val in base_model.items():
        name = key
        for prefix in ("_orig_mod.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name not in hf_sd:
            # constant buffers (e.g. the causal mask) are not in the HF weights;
            # keep the base value, which is identical across steps.
            new_model[key] = base_val
            continue
        w = hf_sd[name]
        if name.endswith(TRANSPOSE):
            w = w.t().contiguous()
        if w.shape != base_val.shape:
            raise SystemExit(
                f"shape mismatch for {name}: mapped {tuple(w.shape)} vs base {tuple(base_val.shape)} "
                f"(transpose={'yes' if name.endswith(TRANSPOSE) else 'no'})"
            )
        new_model[key] = w
        used.add(name)

    missing = set(hf_sd) - used
    if missing:
        # every HF weight should map onto a base key; a leftover means the layouts
        # diverged and the scored model would not be the trained one.
        raise SystemExit(f"HF weights not placed into the model: {sorted(missing)[:8]} ...")

    base["model"] = new_model
    base["iter_num"] = max_step
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    dataset = (base.get("config") or {}).get("dataset")
    print(f"OK {run_dir.name}: max-step {max_step} <- {hf_dir.name}, dataset={dataset!r}, wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
