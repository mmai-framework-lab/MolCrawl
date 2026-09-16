"""Frozen representations from a trained or an untrained trunk.

Used by downstream probes: load a checkpoint, run sequences through the trunk
with the head removed, and read the final hidden states. It lives here rather
than beside one modality's probe so that genome, protein, compounds, rna and
molecule_nat_lang read their numbers out of the same implementation -- with two,
a difference between modalities cannot be told apart from a difference between
implementations.

**Which layer.** The last one, for both families. Nothing earlier is exposed.

**Whether the final normalisation is applied.** Each family keeps its own
convention, because they disagree and neither is wrong:

    nanoGPT   after ``ln_f``. That is what its forward computes before the
              language-model head, and what nanoGPT calls the final hidden state.
    HF        ``last_hidden_state``, the final encoder layer's output. BERT
              applies no further normalisation there.

Forcing one onto the other would change what is measured, so the caller is told
which it got rather than being given a single normalised answer.

**Padding.** Nothing here pads. Every sequence in a batch must already be the
same length, and no attention mask is built, so a masked-out position would be
attended to. A caller with variable-length inputs has to pad and pass its own
mask; this module would silently average padding into any pooled statistic.

**Untrained trunks.** ``untrained=True`` builds the same architecture from the
checkpoint's own configuration and leaves the weights at their initialisation --
the checkpoint supplies the shape and its weights are never read. That is the
floor for an architecture: the same number of features as a trained run, so
whatever a linear classifier reads out of it is not an effect of pretraining.
The initialisation is seeded, and the draw moves the result, so several seeds
are needed before the floor means anything.
"""
from typing import Callable, Tuple

# HF encoders put [CLS] first, so every input token sits one position later than
# it does for a decoder that prepends nothing.
_SPECIAL_PREFIX = {"gpt2": 0, "llama": 0}


def special_token_offset(arch: str) -> int:
    """How far a prepended special token shifts every input position."""
    return _SPECIAL_PREFIX.get(arch, 1)


def strip_compile_prefix(state: dict) -> dict:
    """Drop the ``_orig_mod.`` torch.compile adds; train.py strips it too."""
    pre = "_orig_mod."
    return {(k[len(pre):] if k.startswith(pre) else k): v for k, v in state.items()}


def _nanogpt_encoder(model_path, tok, device, untrained, init_seed):
    """nanoGPT: walk the trunk by hand, since forward() only returns logits."""
    import torch

    from molcrawl.models.gpt2.model import GPT, GPTConfig

    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    margs = dict(ck["model_args"])
    if untrained:
        torch.manual_seed(init_seed)
    model = GPT(GPTConfig(**margs))
    if not untrained:
        model.load_state_dict(strip_compile_prefix(ck["model"]))
    model.to(device).eval()
    tr = model.transformer

    def forward(ids):
        pos = torch.arange(ids.size(1), dtype=torch.long, device=ids.device)
        x = tr.drop(tr.wte(ids) + tr.wpe(pos))
        for block in tr.h:
            x = block(x)
        return tr.ln_f(x)

    def encode(seqs):
        return torch.tensor([tok.convert_tokens_to_ids(list(s)) for s in seqs],
                            dtype=torch.long, device=device)

    return forward, encode, int(margs["n_embd"])


def _hf_encoder(model_path, tok, device, untrained, init_seed):
    """Hugging Face: the trunk is AutoModel, and it wraps the window itself."""
    import torch
    from transformers import AutoConfig, AutoModel

    if untrained:
        torch.manual_seed(init_seed)
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_path))
    else:
        model = AutoModel.from_pretrained(model_path)
    model.to(device).eval()

    def forward(ids):
        return model(input_ids=ids).last_hidden_state

    def encode(seqs):
        cls, sep = tok.cls_token_id, tok.sep_token_id
        return torch.tensor(
            [[cls] + tok.convert_tokens_to_ids(list(s)) + [sep] for s in seqs],
            dtype=torch.long, device=device)

    return forward, encode, int(model.config.hidden_size)


def build_encoder(model_path: str, tokenizer_path: str, arch: str, device: str,
                  untrained: bool = False, init_seed: int = 0
                  ) -> Tuple[Callable, Callable, int]:
    """Return ``(forward, encode, hidden_size)`` for a trunk.

    ``forward(ids)`` gives the final hidden states, ``encode(seqs)`` turns a list
    of character strings into a batch of ids with whatever special tokens the
    family expects. See the module docstring for the layer, the normalisation and
    the padding contract.

    The two families are built by separate helpers rather than by two branches
    here, so that each one's ``forward`` and ``encode`` are named once.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    build = _nanogpt_encoder if arch == "gpt2" else _hf_encoder
    return build(model_path, tok, device, untrained, init_seed)
