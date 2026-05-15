"""GPT-2 decoder adapter for the evaluation framework.

Wraps the minGPT-based :class:`molcrawl.models.gpt2.model.GPT` checkpoint behind
the :class:`~molcrawl.tasks.evaluation._base.model_adapter.ModelAdapter`
interface so evaluators (clinvar, cosmic, omim, MOSES, etc.) stay
architecture-agnostic.

Tokenizer routing is modality-aware: each foundation modality plugs in
the same tokenizer class that its training pipeline uses, so evaluation
reproduces training-time tokenisation byte-for-byte.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

from molcrawl.tasks.evaluation._base.model_adapter import (
    EmbeddingOutput,
    GenerationOutput,
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    register_adapter,
)

logger = logging.getLogger(__name__)

_TOKENIZER_KIND_SPM = "spm"
_TOKENIZER_KIND_HF = "hf"


class GPT2Adapter(ModelAdapter):
    """Adapter for GPT-2 decoders trained with ``molcrawl.models.gpt2``."""

    def __init__(self, handle: ModelHandle):
        super().__init__(handle)
        self.model = None
        self.tokenizer = None
        self._tokenizer_kind: Optional[str] = None
        self._vocab_size: int = 0
        self._torch = None

    def load(self) -> None:
        import torch

        from molcrawl.models.gpt2.model import GPT, GPTConfig

        self._torch = torch

        modality = (self.handle.modality or "").strip() or "genome_sequence"
        self.tokenizer, self._vocab_size, self._tokenizer_kind = self._build_tokenizer(modality)

        device = self.handle.extras.get("device", "cuda")
        logger.info("Loading GPT-2 checkpoint from %s", self.handle.model_path)
        checkpoint = torch.load(self.handle.model_path, map_location=device)
        model_args = checkpoint.get("model_args", {})
        checkpoint_vocab_size = model_args.get("vocab_size")
        if checkpoint_vocab_size is None:
            # Older checkpoints that didn't persist vocab_size: fall back to the
            # tokenizer, matching the adapter's historical behaviour.
            effective_vocab_size = self._vocab_size
        else:
            effective_vocab_size = int(checkpoint_vocab_size)
            if effective_vocab_size != self._vocab_size:
                logger.warning(
                    "Tokenizer vocab_size=%d does not match checkpoint vocab_size=%d; "
                    "using the checkpoint value to match stored weights (token ids "
                    "outside [0, %d) will be out-of-range for the model).",
                    self._vocab_size,
                    effective_vocab_size,
                    effective_vocab_size,
                )
        config = GPTConfig(
            vocab_size=effective_vocab_size,
            block_size=model_args.get("block_size", 1024),
            n_layer=model_args.get("n_layer", 12),
            n_head=model_args.get("n_head", 12),
            n_embd=model_args.get("n_embd", 768),
            dropout=0.0,
            bias=model_args.get("bias", True),
        )
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        self.model = model
        self.device = device

    def _build_tokenizer(self, modality: str) -> Tuple[Any, int, str]:
        """Instantiate the modality-appropriate tokenizer.

        Returns ``(tokenizer, vocab_size, kind)`` where ``kind`` selects
        the encode/decode call shape used by :meth:`_encode`/:meth:`_decode`.
        """
        if modality == "genome_sequence":
            import sentencepiece as spm

            if self.handle.tokenizer_path is None:
                raise ValueError(
                    "GPT2Adapter requires handle.tokenizer_path for modality "
                    "'genome_sequence' (pass the SentencePiece model used during training)."
                )
            logger.info("Loading SentencePiece tokenizer from %s", self.handle.tokenizer_path)
            tok = spm.SentencePieceProcessor(model_file=self.handle.tokenizer_path)
            return tok, int(tok.vocab_size()), _TOKENIZER_KIND_SPM

        if modality == "compounds":
            from molcrawl.data.compounds.utils.tokenizer import CompoundsTokenizer

            vocab_path = self.handle.tokenizer_path or "assets/molecules/vocab.txt"
            logger.info("Loading CompoundsTokenizer from %s", vocab_path)
            tok = CompoundsTokenizer(vocab_path)
            return tok, int(tok.vocab_size), _TOKENIZER_KIND_HF

        if modality == "molecule_nat_lang":
            from molcrawl.data.molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

            # When ``tokenizer_path`` points at an HF tokenizer directory
            # (e.g. a task-specific fine-tune's ``checkpoint-N/`` that bakes
            # its own ``tokenizer.json``), prefer that over the built-in
            # ``MoleculeNatLangTokenizer``. Fine-tune ckpts like
            # ``molecule_nat_lang_mol_instructions/.../checkpoint-N/`` use a
            # different vocab mapping; running them through the master
            # tokenizer otherwise produces token ids the model's embedding
            # table doesn't cover (manifesting as cublasSgemm errors at
            # generation time on the chebi20 evaluator).
            if self.handle.tokenizer_path:
                from pathlib import Path as _Path

                tok_dir = _Path(self.handle.tokenizer_path)
                if tok_dir.is_dir() and (tok_dir / "tokenizer.json").exists():
                    from transformers import AutoTokenizer

                    logger.info(
                        "Loading HF AutoTokenizer for molecule_nat_lang from %s",
                        tok_dir,
                    )
                    tok = AutoTokenizer.from_pretrained(str(tok_dir))
                    vocab_size = (
                        int(tok.vocab_size)
                        if hasattr(tok, "vocab_size")
                        else int(len(tok))
                    )
                    return tok, vocab_size, _TOKENIZER_KIND_HF
                logger.info(
                    "tokenizer_path=%s is not an HF tokenizer dir; "
                    "falling back to built-in MoleculeNatLangTokenizer.",
                    self.handle.tokenizer_path,
                )
            logger.info("Loading built-in MoleculeNatLangTokenizer")
            tok = MoleculeNatLangTokenizer()
            return tok, int(tok.vocab_size), _TOKENIZER_KIND_HF

        if modality == "protein_sequence":
            from molcrawl.data.protein_sequence.dataset.tokenizer import EsmSequenceTokenizer

            if self.handle.tokenizer_path:
                logger.info(
                    "Ignoring --tokenizer-path for modality 'protein_sequence' "
                    "(built-in EsmSequenceTokenizer is used)."
                )
            logger.info("Loading built-in EsmSequenceTokenizer")
            tok = EsmSequenceTokenizer()
            return tok, int(tok.vocab_size), _TOKENIZER_KIND_HF

        if modality == "rna":
            # The rna gpt2 ckpts share the gene-id vocabulary with the rna
            # bert / rnaformer ckpts, so an explicit ``--tokenizer-path``
            # pointing at any saved ``custom_tokenizer_*`` dir works. When
            # the caller doesn't pass one, fall back to the bert tokenizer
            # location (the canonical home of the gene-id WordLevel
            # tokenizer used during rna training).
            from transformers import AutoTokenizer

            tok_dir = self.handle.tokenizer_path
            if tok_dir is None:
                from molcrawl.core.paths import get_custom_tokenizer_path

                tok_dir = get_custom_tokenizer_path("rna", "bert")
            logger.info("Loading rna AutoTokenizer from %s", tok_dir)
            tok = AutoTokenizer.from_pretrained(tok_dir)
            vocab_size = getattr(tok, "vocab_size", None)
            if vocab_size is None:
                vocab_size = len(tok)
            return tok, int(vocab_size), _TOKENIZER_KIND_HF

        raise ValueError(
            f"GPT2Adapter does not yet support modality {modality!r}. "
            "Supported modalities: genome_sequence, compounds, molecule_nat_lang, "
            "protein_sequence, rna. See docs/04-evaluation/tokenizer_paths.md §6."
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None

    def _encode(self, sequence: Any, max_length: Optional[int] = None) -> List[int]:
        # Accept pre-tokenised int lists straight from the parquet pipeline
        # (e.g. rna_benchmark / tabula_sapiens emit token-id lists rather than
        # strings). The HfMlm adapter already short-circuits this case; we
        # need parity here so the same evaluators run on rna gpt2 ckpts.
        tokens: Optional[List[int]] = None
        if isinstance(sequence, (list, tuple)) and len(sequence) > 0:
            try:
                tokens = [int(x) for x in sequence]
            except (TypeError, ValueError):
                tokens = None  # fall through to string tokenisation
        if tokens is None:
            if self._tokenizer_kind == _TOKENIZER_KIND_SPM:
                tokens = list(self.tokenizer.encode(str(sequence)))
            elif self._tokenizer_kind == _TOKENIZER_KIND_HF:
                tokens = list(
                    self.tokenizer.encode(str(sequence), add_special_tokens=False)
                )
            else:
                raise RuntimeError(
                    f"GPT2Adapter has unexpected tokenizer kind {self._tokenizer_kind!r}; "
                    "did load() run?"
                )
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        return list(tokens)

    def _decode(self, ids: List[int]) -> str:
        if self._tokenizer_kind not in (_TOKENIZER_KIND_SPM, _TOKENIZER_KIND_HF):
            raise RuntimeError(
                f"GPT2Adapter has unexpected tokenizer kind {self._tokenizer_kind!r}; "
                "did load() run?"
            )
        # SPM, CompoundsTokenizer (BertTokenizer subclass), MoleculeNatLangTokenizer
        # (TrainableTokenizer wrapper) and EsmSequenceTokenizer all expose a plain
        # decode(ids) method; the wrapper chain in core.base.TrainableTokenizer does
        # not forward keyword arguments, so keep the call shape minimal.
        return self.tokenizer.decode(ids)

    def score_likelihood(
        self, inputs: Sequence[str], context_length: int = 512, **_: Any
    ) -> LikelihoodOutput:
        """Return per-sequence mean log-likelihoods.

        The returned ``log_likelihood`` has shape ``(len(inputs),)`` and
        carries mean per-token log-probabilities, mirroring the existing
        clinvar evaluation.
        """
        if self.model is None:
            raise RuntimeError("GPT2Adapter.load() must be called first")
        torch = self._torch
        import torch.nn.functional as F

        likelihoods: List[float] = []
        num_tokens: List[int] = []
        with torch.no_grad():
            for sequence in inputs:
                tokens = self._encode(sequence, max_length=context_length)
                if len(tokens) == 0:
                    likelihoods.append(0.0)
                    num_tokens.append(0)
                    continue
                ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                dummy_targets = torch.zeros_like(ids)
                logits, _ = self.model(ids, targets=dummy_targets)
                if ids.size(1) <= 1:
                    likelihoods.append(0.0)
                    num_tokens.append(len(tokens))
                    continue
                log_probs = F.log_softmax(logits, dim=-1)
                target_tokens = ids[:, 1:]
                pred_log_probs = log_probs[:, :-1, :]
                token_log_probs = pred_log_probs.gather(
                    2, target_tokens.unsqueeze(2)
                ).squeeze(2)
                likelihoods.append(float(token_log_probs.mean().item()))
                num_tokens.append(len(tokens))

        return LikelihoodOutput(log_likelihood=likelihoods, num_tokens=num_tokens)

    def embed(
        self,
        inputs: Sequence[str],
        pooling: str = "mean",
        context_length: int = 512,
        batch_size: int = 16,
        **_: Any,
    ) -> EmbeddingOutput:
        """Return pooled sequence embeddings from the nanoGPT trunk.

        We re-run the same ``wte + wpe + blocks + ln_f`` stack the decoder
        uses for generation, but stop before the ``lm_head`` projection —
        the post-``ln_f`` activations are what downstream linear probes
        (GUE / TAPE / DeepLoc / Tabula Sapiens / Replogle) consume.

        ``pooling``:
            ``"mean"``  (default) — mean over non-pad token positions.
            ``"last"``  — use the last non-pad position (causal model's
                           "most-informed" token).
            ``"cls"``   — alias for ``"last"`` kept for API symmetry.
        """
        if self.model is None:
            raise RuntimeError("GPT2Adapter.load() must be called first")
        import numpy as np
        import torch

        block_size = getattr(self.model.config, "block_size", None)
        if block_size is not None:
            context_length = min(int(context_length), int(block_size))

        pad_id = 0  # nanoGPT is not pad-aware; we zero out pad positions in the mask.

        pooled_chunks: List[Any] = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = list(inputs[start : start + batch_size])
                token_lists = [
                    self._encode(t, max_length=context_length) for t in batch
                ]
                if not token_lists:
                    continue
                max_len = max((len(tl) for tl in token_lists), default=1)
                max_len = max(max_len, 1)
                padded = [
                    tl + [pad_id] * (max_len - len(tl)) for tl in token_lists
                ]
                mask = [
                    [1] * len(tl) + [0] * (max_len - len(tl)) for tl in token_lists
                ]
                idx = torch.tensor(padded, dtype=torch.long, device=self.device)
                attn = torch.tensor(mask, dtype=torch.float, device=self.device)

                b, t = idx.size()
                pos = torch.arange(0, t, dtype=torch.long, device=self.device)
                tok_emb = self.model.transformer.wte(idx)
                pos_emb = self.model.transformer.wpe(pos)
                x = self.model.transformer.drop(tok_emb + pos_emb)
                for block in self.model.transformer.h:
                    x = block(x)
                x = self.model.transformer.ln_f(x)  # (B, L, D)

                if pooling == "mean":
                    weight = attn.unsqueeze(-1)
                    pooled = (x * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)
                elif pooling in ("last", "cls"):
                    # Pick the last non-pad index per row.
                    last_idx = attn.long().sum(dim=1) - 1  # (B,)
                    last_idx = last_idx.clamp(min=0)
                    pooled = x[torch.arange(b, device=self.device), last_idx]
                else:
                    raise ValueError(
                        f"Unsupported pooling={pooling!r}; choose 'mean' / 'last' / 'cls'."
                    )

                pooled_chunks.append(pooled.cpu().numpy())

        if not pooled_chunks:
            return EmbeddingOutput(embeddings=np.zeros((0, 0)), pooled=True)
        return EmbeddingOutput(
            embeddings=np.concatenate(pooled_chunks, axis=0), pooled=True
        )

    def generate(
        self,
        prompts: Optional[Sequence[str]] = None,
        num_samples: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        **_: Any,
    ) -> GenerationOutput:
        if self.model is None:
            raise RuntimeError("GPT2Adapter.load() must be called first")
        torch = self._torch

        prompt_list = list(prompts) if prompts else [""]
        # Fallback start-of-sequence token when the user requests free
        # generation (no prompt). Token id 0 happens to be [PAD] for the
        # CompoundsTokenizer / BertTokenizer trained checkpoints, which
        # makes the model generate further pad tokens; prefer the
        # tokenizer's BOS or CLS when one is registered so the generation
        # starts from a real "begin-of-sequence" anchor.
        default_start_id = 0
        if self._tokenizer_kind == _TOKENIZER_KIND_HF and self.tokenizer is not None:
            default_start_id = (
                getattr(self.tokenizer, "bos_token_id", None)
                or getattr(self.tokenizer, "cls_token_id", None)
                or 0
            )
        # The underlying GPT-2 model.generate divides logits by temperature;
        # passing temperature == 0 yields inf logits and softmax NaN. Floor
        # to a small epsilon so temperature=0 behaves as greedy decoding via
        # a peaked softmax instead of crashing.
        eff_temp = max(float(temperature), 1e-5)

        sequences: List[str] = []
        with torch.no_grad():
            for prompt in prompt_list:
                for _ in range(num_samples):
                    prompt_ids = self._encode(prompt) or [default_start_id]
                    idx = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                    out = self.model.generate(
                        idx,
                        max_new_tokens=max_new_tokens,
                        temperature=eff_temp,
                        top_k=top_k,
                    )
                    decoded = self._decode(out[0].tolist())
                    sequences.append(decoded)
        return GenerationOutput(
            sequences=sequences,
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "num_samples": num_samples,
            },
        )


register_adapter("gpt2", GPT2Adapter)
