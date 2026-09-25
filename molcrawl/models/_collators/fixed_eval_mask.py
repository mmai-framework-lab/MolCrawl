"""Give evaluation the same masked positions every time it runs.

An MLM collator masks at collate time from the global torch RNG, so each evaluation
masks a different draw of positions. Two evaluations of the same model then differ by
the draw as well as by the model, and on the molecule_nat_lang grid that draw moved the
number by about as much as the gap between two learning rates: the last ten evaluations
of a converged run spread 0.0074 to 0.0159, against a 0.0062 gap between the best small
and the best medium arm (2026-09-24 report).

This wrapper reseeds the RNG from the batch itself before handing it to the collator,
then puts the RNG back. Same rows, same masked positions, whatever else the process has
drawn since -- across evaluations, across segments of a run, and across runs that share
the seed.

Deriving the seed from the batch rather than counting calls is what makes it hold with
dataloader workers. Each worker gets its own copy of the collator, so a counter would
restart in each of them and different batches would share a seed; the batch's own bytes
do not depend on which worker picked it up.

Training is left alone: main.py wraps only the collator the evaluation dataloader uses,
so no gradient is computed from a mask this file chose.
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, List

import torch


def _batch_key(features: List[Dict[str, Any]]) -> int:
    """A number that depends on the batch's contents and nothing else."""
    digest = 0
    for feature in features:
        ids = feature["input_ids"] if isinstance(feature, dict) else feature
        if torch.is_tensor(ids):
            ids = ids.detach().to("cpu", torch.int64)
            payload = ids.numpy().tobytes()
        else:
            payload = torch.tensor(list(ids), dtype=torch.int64).numpy().tobytes()
        digest = zlib.crc32(payload, digest)
    return digest


class FixedEvalMaskCollator:
    """Wrap a collator so a given batch is always masked the same way."""

    def __init__(self, base_collator, seed: int = 42):
        self.base_collator = base_collator
        self.seed = int(seed)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        key = (_batch_key(features) ^ (self.seed & 0xFFFFFFFF)) & 0xFFFFFFFF
        cpu_state = torch.get_rng_state()
        # torch.manual_seed seeds every CUDA device as well, so put those back too --
        # but only if CUDA is already up. Asking otherwise would initialise it inside a
        # dataloader worker, which is not this wrapper's business.
        cuda_states = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() and torch.cuda.is_initialized() else None)
        torch.manual_seed(key)
        try:
            return self.base_collator(features)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
