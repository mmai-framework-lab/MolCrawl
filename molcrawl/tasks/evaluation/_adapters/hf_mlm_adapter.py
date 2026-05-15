"""Unified HuggingFace MaskedLM adapter.

Handles any HF-format MLM checkpoint produced by ``molcrawl.models.bert``,
``molcrawl.models.esm2``, ``molcrawl.models.chemberta2``, ``molcrawl.models.dnabert2``, or
``molcrawl.models.rnaformer``. Weights are loaded through
``AutoModelForMaskedLM.from_pretrained``, so the concrete model class
(``BertForMaskedLM``, ``EsmForMaskedLM``, ``RobertaForMaskedLM``, ...)
is resolved from the checkpoint's ``config.json``.

Tokenizer resolution walks three tiers:

1. ``--tokenizer-path`` via ``AutoTokenizer.from_pretrained`` (if
   provided).
2. The checkpoint directory itself — HF Trainer co-saves
   ``tokenizer.json`` / ``special_tokens_map.json`` /
   ``tokenizer_config.json`` next to ``model.safetensors`` for the BERT
   modalities that train end-to-end with a new tokenizer; other runs
   skip this step.
3. An ``(arch, modality)``-aware fallback that mirrors the training-time
   wiring (``CompoundsTokenizer``, ``MoleculeNatLangTokenizer``,
   ``BertProteinSequenceTokenizer``, or the
   ``custom_tokenizer_<arch>/`` directory saved under
   ``LEARNING_SOURCE_DIR`` for DNABERT-2 and RNAformer).

``score_likelihood`` is pseudo-log-likelihood (PLL): every non-special
position is masked in turn, the MLM head is evaluated, and the mean
log-probability of the original token is returned. Positions are batched
in chunks of 64 (overridable via ``ModelHandle.extras["bert_mlm_batch"]``)
to keep device memory bounded on long sequences.

``generate`` is not supported — MLM adapters are encoder-only; use
``arch=gpt2`` for generation tasks.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from molcrawl.tasks.evaluation._base.model_adapter import (
    EmbeddingOutput,
    GenerationOutput,
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    register_adapter,
)

logger = logging.getLogger(__name__)

_SUPPORTED_ARCHS = ("bert", "esm2", "chemberta2", "dnabert2", "rnaformer")

# HF repos whose canonical loader path ships custom modeling code that
# AutoModelForMaskedLM can only execute when ``trust_remote_code=True``.
# Matching is substring-based on the model_path so both ``org/repo`` and
# locally-cloned copies trigger the flag.
_TRUSTED_REMOTE_CODE_HINTS = (
    "DNABERT-2",
    "DNABERT2",
    "dnabert-2",
    "RNAformer",
    "rnaformer",
    "InstaDeepAI",
    "NucleotideTransformer",
)


def _looks_like_trusted_remote_code_repo(model_path: str) -> bool:
    if not isinstance(model_path, str) or not model_path:
        return False
    return any(hint in model_path for hint in _TRUSTED_REMOTE_CODE_HINTS)


def _np_integer():
    """Return numpy.integer if numpy is importable, else a sentinel never matched."""
    try:
        import numpy as _np

        return _np.integer
    except ImportError:  # pragma: no cover
        class _NoMatch:
            pass

        return _NoMatch


class HfMlmAdapter(ModelAdapter):
    """Adapter for any HuggingFace-format MLM checkpoint."""

    def __init__(self, handle: ModelHandle):
        super().__init__(handle)
        self.model = None
        self.tokenizer = None
        self._torch = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForMaskedLM

        self._torch = torch
        device = self.handle.extras.get("device", "cuda")

        self.tokenizer = self._load_tokenizer()
        logger.info(
            "Loading HF MLM checkpoint from %s (arch=%s, modality=%s)",
            self.handle.model_path,
            self.handle.arch,
            self.handle.modality,
        )
        # Auto-enable trust_remote_code for community HF models that ship
        # custom modeling code (DNABERT-2, RNAformer, etc.). The flag can
        # also be set explicitly via ``extras.trust_remote_code`` or the
        # ``TRUST_REMOTE_CODE=1`` env var.
        import os
        trust_remote_code = (
            bool(self.handle.extras.get("trust_remote_code", False))
            or os.environ.get("TRUST_REMOTE_CODE", "").strip() == "1"
            or _looks_like_trusted_remote_code_repo(self.handle.model_path)
        )
        model = AutoModelForMaskedLM.from_pretrained(
            self.handle.model_path, trust_remote_code=trust_remote_code
        )
        # Stash for the tokenizer loader.
        self._trust_remote_code = trust_remote_code
        model.to(device)
        model.eval()
        self.model = model
        self.device = device

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tried: List[str] = []

        # 1. Explicit --tokenizer-path wins.
        if self.handle.tokenizer_path:
            tried.append(str(self.handle.tokenizer_path))
            try:
                logger.info(
                    "Loading tokenizer from --tokenizer-path %s",
                    self.handle.tokenizer_path,
                )
                return AutoTokenizer.from_pretrained(
                    self.handle.tokenizer_path,
                    trust_remote_code=getattr(self, "_trust_remote_code", False),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AutoTokenizer.from_pretrained(%r) failed: %s. Falling back.",
                    self.handle.tokenizer_path,
                    exc,
                )

        # 2. Checkpoint directory (HF Trainer co-saves tokenizer for some archs).
        tried.append(str(self.handle.model_path))
        try:
            logger.info(
                "Trying tokenizer co-saved with checkpoint at %s",
                self.handle.model_path,
            )
            return AutoTokenizer.from_pretrained(
                self.handle.model_path,
                trust_remote_code=getattr(self, "_trust_remote_code", False),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "Checkpoint %r does not carry a tokenizer (%s); falling back to "
                "arch+modality defaults.",
                self.handle.model_path,
                exc,
            )

        # 3. (arch, modality) fallback.
        return self._arch_modality_fallback(tried)

    def _arch_modality_fallback(self, tried: List[str]):
        arch = (self.handle.arch or "").strip()
        modality = (self.handle.modality or "").strip() or "genome_sequence"
        key = (arch, modality)

        if arch in ("bert", "chemberta2") and modality == "compounds":
            from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer

            logger.info(
                "%s/%s fallback: CompoundsTokenizer(assets/molecules/vocab.txt)",
                arch,
                modality,
            )
            return CompoundsTokenizer("assets/molecules/vocab.txt")

        if arch == "bert" and modality == "molecule_nat_lang":
            from molcrawl.data.molecule_nat_lang.utils.tokenizer import (
                MoleculeNatLangTokenizer,
            )

            logger.info("%s/%s fallback: MoleculeNatLangTokenizer", arch, modality)
            return MoleculeNatLangTokenizer()

        if arch in ("bert", "esm2") and modality == "protein_sequence":
            from molcrawl.data.protein_sequence.utils.bert_tokenizer import (
                BertProteinSequenceTokenizer,
            )

            logger.info(
                "%s/%s fallback: BertProteinSequenceTokenizer", arch, modality
            )
            return BertProteinSequenceTokenizer()

        if key in (
            ("bert", "genome_sequence"),
            ("bert", "rna"),
            ("dnabert2", "genome_sequence"),
            ("rnaformer", "rna"),
        ):
            from transformers import AutoTokenizer
            from molcrawl.core.paths import get_custom_tokenizer_path

            tok_dir = get_custom_tokenizer_path(modality, arch)
            tried.append(tok_dir)
            logger.info(
                "%s/%s fallback: AutoTokenizer.from_pretrained(%s)",
                arch,
                modality,
                tok_dir,
            )
            return AutoTokenizer.from_pretrained(
                tok_dir,
                trust_remote_code=getattr(self, "_trust_remote_code", False),
            )

        raise ValueError(
            f"HfMlmAdapter has no tokenizer fallback for arch={arch!r} "
            f"modality={modality!r}. Tried: {tried}. Pass --tokenizer-path "
            "pointing at an HF tokenizer directory."
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None

    def _encode(self, sequence: str, max_length: Optional[int] = None) -> List[int]:
        ids = list(self.tokenizer.encode(sequence, add_special_tokens=True))
        if max_length is not None and len(ids) > max_length:
            ids = ids[: max_length - 1] + [ids[-1]]
        return ids

    def score_likelihood(
        self, inputs: Sequence[Any], context_length: int = 512, **_: Any
    ) -> LikelihoodOutput:
        """Return per-sequence mean pseudo-log-likelihoods.

        ``inputs`` may be either a sequence of strings (each gets encoded
        through ``self.tokenizer``) or a sequence of pre-tokenised int
        lists (each is treated as the sequence's token-id stream
        directly). The pre-tokenised path is used by
        ``rna_benchmark`` so that the adapter does not need to re-encode
        cells whose original tokenisation lives in parquet/JSONL.
        """
        if self.model is None:
            raise RuntimeError("HfMlmAdapter.load() must be called first")
        torch = self._torch
        import torch.nn.functional as F

        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None:
            raise RuntimeError(
                "HfMlmAdapter.score_likelihood requires a tokenizer with a "
                "mask_token_id; got None."
            )
        special_ids = set(getattr(self.tokenizer, "all_special_ids", ()) or ())

        cfg = getattr(self.model, "config", None)
        max_pos = (
            getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
        )
        if max_pos is not None:
            context_length = min(int(context_length), max(1, int(max_pos) - 2))

        mlm_batch_size = int(self.handle.extras.get("bert_mlm_batch", 64))

        likelihoods: List[float] = []
        num_tokens: List[int] = []
        with torch.no_grad():
            for sequence in inputs:
                if isinstance(sequence, (list, tuple)) and (
                    not sequence or isinstance(sequence[0], (int, _np_integer()))
                ):
                    ids = [int(t) for t in sequence]
                    if context_length is not None and len(ids) > context_length:
                        ids = ids[:context_length]
                else:
                    ids = self._encode(sequence, max_length=context_length)
                if len(ids) == 0:
                    likelihoods.append(0.0)
                    num_tokens.append(0)
                    continue

                positions = [i for i, tok in enumerate(ids) if tok not in special_ids]
                if not positions:
                    likelihoods.append(0.0)
                    num_tokens.append(len(ids))
                    continue

                base = torch.tensor(ids, dtype=torch.long, device=self.device)
                total_logp = 0.0
                for start in range(0, len(positions), mlm_batch_size):
                    pos_chunk = positions[start : start + mlm_batch_size]
                    batch = base.unsqueeze(0).repeat(len(pos_chunk), 1).clone()
                    for row, pos in enumerate(pos_chunk):
                        batch[row, pos] = mask_id
                    attn = torch.ones_like(batch)
                    logits = self.model(input_ids=batch, attention_mask=attn).logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    for row, pos in enumerate(pos_chunk):
                        total_logp += float(log_probs[row, pos, ids[pos]].item())

                likelihoods.append(total_logp / len(positions))
                num_tokens.append(len(positions))

        return LikelihoodOutput(log_likelihood=likelihoods, num_tokens=num_tokens)

    def embed(
        self,
        inputs: Sequence[Any],
        pooling: str = "mean",
        context_length: int = 512,
        batch_size: int = 16,
        **_: Any,
    ) -> EmbeddingOutput:
        """Return pooled sequence embeddings from the MLM backbone.

        ``inputs`` may be either a sequence of strings (each gets encoded
        through ``self.tokenizer``) or a sequence of pre-tokenised int
        lists (each is treated as the sequence's token-id stream
        directly). The pre-tokenised path is used by the rna evaluators
        that ship token ids in JSONL — same convention as
        :meth:`score_likelihood` — so we don't need a tokenizer
        round-trip when the upstream pipeline already produced ids.

        Drops the MLM projection head and exposes ``last_hidden_state``
        via ``base_model``. ``pooling``:

        - ``"mean"`` (default): attention-mask-weighted average.
        - ``"cls"``:  first-token representation ([CLS] / <cls>, standard
                     BERT / RoBERTa convention).
        """
        if self.model is None:
            raise RuntimeError("HfMlmAdapter.load() must be called first")
        import numpy as np
        import torch

        encoder = getattr(self.model, "base_model", self.model)

        # Respect the model's positional-embedding ceiling. RoBERTa /
        # ChemBERTa-2 use a small max (258) and reserve 2 slots for the
        # padding-id offset, so we subtract a buffer of 2 before encoding
        # to avoid `position_id >= max_position_embeddings` errors on
        # long inputs (e.g. HIV SMILES that tokenise to 300+ tokens).
        cfg = getattr(self.model, "config", None)
        max_pos = getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
        if max_pos is not None:
            context_length = min(int(context_length), max(1, int(max_pos) - 2))

        pooled_chunks: List[Any] = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = list(inputs[start : start + batch_size])
                token_lists: List[List[int]] = []
                for t in batch:
                    if isinstance(t, (list, tuple)) and (
                        not t or isinstance(t[0], (int, _np_integer()))
                    ):
                        ids = [int(x) for x in t]
                        if context_length is not None and len(ids) > context_length:
                            ids = ids[:context_length]
                        token_lists.append(ids)
                    else:
                        token_lists.append(
                            self._encode(t, max_length=context_length)
                        )
                if not token_lists:
                    continue

                pad_id = (
                    getattr(self.tokenizer, "pad_token_id", None)
                    if self.tokenizer is not None
                    else None
                )
                if pad_id is None:
                    pad_id = 0

                max_len = max((len(tl) for tl in token_lists), default=1)
                max_len = max(max_len, 1)
                padded = [
                    tl + [pad_id] * (max_len - len(tl)) for tl in token_lists
                ]
                mask = [
                    [1] * len(tl) + [0] * (max_len - len(tl)) for tl in token_lists
                ]
                idx = torch.tensor(padded, dtype=torch.long, device=self.device)
                attn = torch.tensor(mask, dtype=torch.long, device=self.device)

                outputs = encoder(input_ids=idx, attention_mask=attn)
                last_hidden = outputs.last_hidden_state  # (B, L, D)

                if pooling == "mean":
                    weight = attn.float().unsqueeze(-1)
                    pooled = (last_hidden * weight).sum(dim=1) / weight.sum(
                        dim=1
                    ).clamp(min=1)
                elif pooling in ("cls", "first"):
                    pooled = last_hidden[:, 0, :]
                else:
                    raise ValueError(
                        f"Unsupported pooling={pooling!r}; choose 'mean' / 'cls'."
                    )
                pooled_chunks.append(pooled.cpu().numpy())

        if not pooled_chunks:
            return EmbeddingOutput(embeddings=np.zeros((0, 0)), pooled=True)
        return EmbeddingOutput(
            embeddings=np.concatenate(pooled_chunks, axis=0), pooled=True
        )

    def embed_per_residue(
        self,
        inputs: Sequence[Any],
        context_length: int = 512,
        batch_size: int = 8,
        **_: Any,
    ) -> List[Any]:
        """Return one ``(L_i, D)`` array per input — no pooling.

        ``inputs`` may be either strings or pre-tokenised int lists, the
        same as :meth:`embed`. Special tokens (cls / sep / pad) are
        kept in the per-position output; the caller is responsible for
        masking them when training a residue-level probe.

        Used by sequence-labeling tasks (TAPE secondary_structure_*)
        that need positional features rather than a single pooled
        embedding.
        """
        if self.model is None:
            raise RuntimeError("HfMlmAdapter.load() must be called first")
        import numpy as np
        import torch

        encoder = getattr(self.model, "base_model", self.model)

        cfg = getattr(self.model, "config", None)
        max_pos = (
            getattr(cfg, "max_position_embeddings", None) if cfg is not None else None
        )
        if max_pos is not None:
            context_length = min(int(context_length), max(1, int(max_pos) - 2))

        out_per_input: List[Any] = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = list(inputs[start : start + batch_size])
                token_lists: List[List[int]] = []
                for t in batch:
                    if isinstance(t, (list, tuple)) and (
                        not t or isinstance(t[0], (int, _np_integer()))
                    ):
                        ids = [int(x) for x in t]
                        if context_length is not None and len(ids) > context_length:
                            ids = ids[:context_length]
                        token_lists.append(ids)
                    else:
                        token_lists.append(
                            self._encode(t, max_length=context_length)
                        )
                if not token_lists:
                    continue

                pad_id = (
                    getattr(self.tokenizer, "pad_token_id", None)
                    if self.tokenizer is not None
                    else None
                )
                if pad_id is None:
                    pad_id = 0

                lengths = [len(tl) for tl in token_lists]
                max_len = max(lengths)
                padded = [
                    tl + [pad_id] * (max_len - len(tl)) for tl in token_lists
                ]
                mask = [
                    [1] * len(tl) + [0] * (max_len - len(tl)) for tl in token_lists
                ]
                idx = torch.tensor(padded, dtype=torch.long, device=self.device)
                attn = torch.tensor(mask, dtype=torch.long, device=self.device)

                outputs = encoder(input_ids=idx, attention_mask=attn)
                last_hidden = outputs.last_hidden_state  # (B, L, D)
                last_hidden_np = last_hidden.cpu().numpy()
                for row_i, L in enumerate(lengths):
                    out_per_input.append(last_hidden_np[row_i, :L, :].copy())
        return out_per_input

    def generate(
        self,
        prompts: Optional[Sequence[str]] = None,
        num_samples: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        **_: Any,
    ) -> GenerationOutput:
        raise NotImplementedError(
            "HfMlmAdapter does not support autoregressive generation; "
            "MLM adapters are encoder-only. Use arch=gpt2 for generation tasks."
        )


for _arch in _SUPPORTED_ARCHS:
    register_adapter(_arch, HfMlmAdapter)
