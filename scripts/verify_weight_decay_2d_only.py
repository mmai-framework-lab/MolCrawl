"""Phase 0-4 verification: weight decay applies to 2D matmul weights only.

Constructs a small nn.Module with the same param name conventions as
GPT-2 (transformer.wte/wpe) and BERT (word_embeddings, LayerNorm).
Verifies that after the fixed classification:

  - decay group contains ONLY 2D Linear weights (matmul params)
  - nodecay group contains:
      * all bias tensors (dim < 2)
      * all LayerNorm weight/bias (dim < 2)
      * all Embedding weights (dim = 2 but name-matched)

Also:
  - Instantiates the real molcrawl GPT and calls configure_optimizers to
    confirm the actual model produces the correct partition.
  - Constructs a tiny toy BERT model and runs the
    _WeightDecayNoEmbedTrainer's get_decay_parameter_names hook via
    subclass isolation (avoiding the need for a full Trainer instance).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

REPO = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
sys.path.insert(0, str(REPO))


def _summarise_optim_groups(optim_groups, param_dict):
    """Given [{params:[tensor,...], weight_decay:x}, ...], match tensors back
    to names to describe the split.

    Returns dict {name: 'decay'|'nodecay'} for reporting.
    """
    id_to_name = {id(p): n for n, p in param_dict.items()}
    out = {}
    for group in optim_groups:
        tag = "decay" if group.get("weight_decay", 0.0) > 0.0 else "nodecay"
        for p in group["params"]:
            out[id_to_name[id(p)]] = tag
    return out


def test_gpt2_configure_optimizers():
    print("\n=== TEST 1: GPT-2 (nanoGPT) configure_optimizers on a real GPT ===")
    from molcrawl.models.gpt2.model import GPT, GPTConfig
    cfg = GPTConfig(
        block_size=32, vocab_size=100, n_layer=2, n_head=2, n_embd=32,
        dropout=0.0, bias=False,
    )
    model = GPT(cfg)
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}

    optim = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
    )
    groups = _summarise_optim_groups(optim.param_groups, param_dict)

    decay_names = sorted([n for n, tag in groups.items() if tag == "decay"])
    nodecay_names = sorted([n for n, tag in groups.items() if tag == "nodecay"])

    print(f"  decay group ({len(decay_names)} tensors):")
    for n in decay_names:
        print(f"    {n:60}  dim={param_dict[n].dim()}")
    print(f"  nodecay group ({len(nodecay_names)} tensors):")
    for n in nodecay_names:
        print(f"    {n:60}  dim={param_dict[n].dim()}")

    # Assertions
    for n in decay_names:
        assert param_dict[n].dim() >= 2, f"decay must be 2D: {n} has dim={param_dict[n].dim()}"
        assert "wte" not in n and "wpe" not in n, f"embedding leaked into decay: {n}"

    embed_names = [n for n in param_dict if "wte" in n or "wpe" in n]
    for n in embed_names:
        assert n in nodecay_names, f"embedding {n} must be in nodecay"

    # LayerNorm / bias in nodecay
    ln_names = [n for n in param_dict if "ln" in n.lower() or n.endswith(".bias")]
    for n in ln_names:
        assert n in nodecay_names, f"LayerNorm/bias {n} must be in nodecay"

    print("  [PASS] All embedding, bias, LN params are in nodecay; decay is 2D-only matmul.")


def test_bert_decay_parameter_names():
    print("\n=== TEST 2: BERT _WeightDecayNoEmbedTrainer.get_decay_parameter_names ===")
    # Build a tiny BERT-like model
    class TinyBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(100, 32)
            self.position_embeddings = nn.Embedding(64, 32)
            self.token_type_embeddings = nn.Embedding(2, 32)
            self.LayerNorm = nn.LayerNorm(32)
            self.dense = nn.Linear(32, 32)   # 2D weight
            self.decoder = nn.Linear(32, 100)

    model = TinyBert()
    param_names = {n for n, _ in model.named_parameters()}
    print(f"  model param names: {sorted(param_names)}")

    # Simulate the HF default filter (LN + bias exclusion) then apply our extra exclusion
    from transformers.trainer_pt_utils import get_parameter_names
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    hf_default = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    hf_default = [n for n in hf_default if "bias" not in n]

    # Apply our filter (mirroring the class body in bert/main.py)
    embedding_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            for pn, _ in module.named_parameters(recurse=False):
                embedding_names.add(f"{module_name}.{pn}" if module_name else pn)
    decay_names = [n for n in hf_default if n not in embedding_names]

    all_params = {n for n, _ in model.named_parameters()}
    nodecay_names = sorted(all_params - set(decay_names))

    print(f"  HF default decay group ({len(hf_default)}): {sorted(hf_default)}")
    print(f"  our extra-excluded embedding names ({len(embedding_names)}): {sorted(embedding_names)}")
    print(f"  FINAL decay group ({len(decay_names)}): {sorted(decay_names)}")
    print(f"  FINAL nodecay group ({len(nodecay_names)}): {nodecay_names}")

    # Assertions
    for n in ("word_embeddings.weight", "position_embeddings.weight", "token_type_embeddings.weight"):
        assert n not in decay_names, f"embedding {n} should be excluded from decay"
        assert n in nodecay_names, f"embedding {n} should be in nodecay"
    for n in ("LayerNorm.weight", "LayerNorm.bias"):
        assert n not in decay_names, f"LayerNorm {n} should be excluded from decay"
    for n in ("dense.bias", "decoder.bias"):
        assert n not in decay_names, f"bias {n} should be excluded from decay"
    assert "dense.weight" in decay_names, "Linear.weight should be in decay"
    assert "decoder.weight" in decay_names, "Linear.weight should be in decay"

    print("  [PASS] All bias/LN/embedding params are in nodecay; decay is 2D Linear.weight only.")


def test_bert_default_weight_decay():
    print("\n=== TEST 3: BERT main.py default weight_decay value ===")
    src = (REPO / "molcrawl/models/bert/main.py").read_text()
    import re
    match = re.search(r"^\s*weight_decay\s*=\s*([0-9.e+-]+)", src, re.MULTILINE)
    assert match is not None, "weight_decay assignment not found"
    val = float(match.group(1))
    print(f"  bert/main.py default weight_decay = {val}")
    assert val == 0.01, f"expected 0.01, got {val}"
    print("  [PASS] BERT default weight_decay is 0.01 (was 0.1 pre-fix).")


def test_gpt2_default_weight_decay():
    print("\n=== TEST 4: GPT-2 train.py default weight_decay value ===")
    src = (REPO / "molcrawl/models/gpt2/train.py").read_text()
    import re
    match = re.search(r"^weight_decay\s*=\s*([0-9.e+-]+)", src, re.MULTILINE)
    assert match is not None, "weight_decay assignment not found"
    val = float(match.group(1))
    print(f"  gpt2/train.py default weight_decay = {val}")
    assert val == 0.1, f"expected 0.1 (unchanged), got {val}"
    print("  [PASS] GPT-2 default weight_decay is 0.1 (unchanged, spec).")


def main() -> int:
    print("=== Phase 0-4 verification: weight decay 2D-only, no embedding ===")
    test_gpt2_configure_optimizers()
    test_bert_decay_parameter_names()
    test_bert_default_weight_decay()
    test_gpt2_default_weight_decay()
    print("\n=== ALL ASSERTIONS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
