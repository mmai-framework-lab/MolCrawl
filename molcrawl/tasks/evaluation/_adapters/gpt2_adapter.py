"""GPT-2 decoder adapter for the evaluation framework.

Wraps the minGPT-based :class:`molcrawl.gpt2.model.GPT` checkpoint behind
the :class:`~molcrawl.tasks.evaluation._base.model_adapter.ModelAdapter`
interface so evaluators (clinvar, cosmic, omim, MOSES, etc.) stay
architecture-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from molcrawl.tasks.evaluation._base.model_adapter import (
    GenerationOutput,
    LikelihoodOutput,
    ModelAdapter,
    ModelHandle,
    register_adapter,
)

logger = logging.getLogger(__name__)


class GPT2Adapter(ModelAdapter):
    """Adapter for GPT-2 decoders trained with ``molcrawl.gpt2``."""

    def __init__(self, handle: ModelHandle):
        super().__init__(handle)
        self.model = None
        self.tokenizer = None
        self._torch = None

    def load(self) -> None:
        import sentencepiece as spm
        import torch

        from molcrawl.gpt2.model import GPT, GPTConfig

        self._torch = torch

        if self.handle.tokenizer_path is None:
            raise ValueError(
                "GPT2Adapter requires handle.tokenizer_path to point at the "
                "SentencePiece model used during training."
            )

        logger.info("Loading GPT-2 tokenizer from %s", self.handle.tokenizer_path)
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=self.handle.tokenizer_path
        )

        device = self.handle.extras.get("device", "cuda")
        logger.info("Loading GPT-2 checkpoint from %s", self.handle.model_path)
        checkpoint = torch.load(self.handle.model_path, map_location=device)
        model_args = checkpoint.get("model_args", {})
        config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size(),
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

    def close(self) -> None:
        self.model = None
        self.tokenizer = None

    def _encode(self, sequence: str, max_length: Optional[int] = None) -> List[int]:
        tokens = self.tokenizer.encode(sequence)
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

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
        sequences: List[str] = []
        with torch.no_grad():
            for prompt in prompt_list:
                for _ in range(num_samples):
                    prompt_ids = self.tokenizer.encode(prompt) or [0]
                    idx = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                    out = self.model.generate(
                        idx,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    decoded = self.tokenizer.decode(out[0].tolist())
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
