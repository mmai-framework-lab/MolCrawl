"""Compatibility shim for ``torch >= 2.6`` with HuggingFace Trainer resume.

PyTorch 2.6 flipped the default of ``torch.load(weights_only=...)`` from
``False`` to ``True``. ``transformers.Trainer._load_rng_state`` reloads
the ``rng_state.pth`` saved during ``Trainer.train`` without passing
``weights_only`` explicitly, so it inherits the new safe default and
refuses to unpickle the ``numpy`` arrays returned by
``numpy.random.get_state()``::

    _pickle.UnpicklingError: Weights only load failed.
    WeightsUnpickler error: Unsupported global:
        GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default.
    ... or numpy.dtypes.UInt32DType ... or any other numpy primitive.

The numpy class surface that needs allowlisting keeps growing per
torch/numpy release, so instead we restore the pre-2.6 default by
monkey-patching ``torch.load`` to default ``weights_only=False`` when the
caller does not specify it. Callers that *do* set ``weights_only=True``
explicitly are unaffected.

This is appropriate for this project because every ``torch.load`` call
in molcrawl is reading checkpoints, optimizer state, or rng state that
we ourselves produced — i.e. trusted sources. Drop the shim once HF
Trainer threads ``weights_only=False`` through ``_load_rng_state`` (or
once all checkpoints move to safetensors).

Affected entry points (each calls ``enable_full_torch_load`` once at
import time):

- ``molcrawl/models/bert/main.py``
- ``molcrawl/models/chemberta2/main.py``
- ``molcrawl/models/dnabert2/main.py``
- ``molcrawl/models/esm2/main.py``
- ``molcrawl/models/main.py``
- ``molcrawl/models/gpt2/train.py``
- ``molcrawl/models/llama/train.py``

The last two do not go through HF Trainer, but they are affected all the
same and for the same reason: they save ``numpy.random.get_state()`` into
each checkpoint for exact resume, and their ``torch.load`` calls pass only
``map_location``. This docstring previously claimed GPT-2 loaded "with
explicit ``weights_only=False`` semantics" — it never did, and the gap
only surfaced when RNA gpt2-xl became the first run long enough to need a
resume (job 28851, 2026-08-17).
"""

from __future__ import annotations


def enable_full_torch_load() -> None:
    """Restore pre-torch-2.6 default of ``weights_only=False`` for ``torch.load``.

    Idempotent. No-op if torch is unavailable. Only changes the default —
    callers that explicitly pass ``weights_only=True`` are unaffected.
    """
    try:
        import torch
    except Exception:
        return

    original = torch.load
    if getattr(original, "_molcrawl_weights_only_patched", False):
        return  # already patched in this interpreter

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    _load._molcrawl_weights_only_patched = True  # type: ignore[attr-defined]
    _load.__wrapped__ = original  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]
