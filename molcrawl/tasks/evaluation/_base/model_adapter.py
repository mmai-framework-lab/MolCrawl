"""Uniform model adapter for evaluation tasks.

The goal of the adapter layer is to isolate evaluation logic from the
specific architecture (GPT-2 decoder, BERT-style encoder, ChemBERTa-2,
ESM-2, DNABERT-2, RNAformer).  Each concrete adapter translates between a
task-level request (``predict_labels``, ``embed``, ``score_likelihood``,
``generate``) and the underlying model API.

Only a minimal, architecture-agnostic surface is declared here.  Concrete
adapters live alongside the foundation model packages (for example
``molcrawl.gpt2.adapter``) and register themselves with
:func:`register_adapter` when imported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


@dataclass
class ModelHandle:
    """Serialisable description of a trained model on disk.

    Attributes:
        arch: Architecture tag, one of ``{"gpt2", "bert", "chemberta2",
            "esm2", "dnabert2", "rnaformer"}``.
        modality: Foundation-model modality, one of ``{"genome_sequence",
            "protein_sequence", "compounds", "rna", "molecule_nat_lang"}``.
        model_path: Absolute path to the checkpoint or model directory.
        tokenizer_path: Absolute path to the tokenizer file, when the
            architecture uses an external tokenizer.  ``None`` when the
            model ships with a built-in tokenizer (for example ESM-2).
        size: Optional size tag (``"small" | "medium" | "large" | "xl"``).
        extras: Free-form key/value metadata specific to the arch.
    """

    arch: str
    modality: str
    model_path: str
    tokenizer_path: Optional[str] = None
    size: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationOutput:
    """Classification prediction output for one or more inputs."""

    logits: Any  # numpy.ndarray or torch.Tensor, shape (B, C)
    labels: Optional[Any] = None  # predicted argmax, shape (B,)
    probabilities: Optional[Any] = None  # softmax, shape (B, C)


@dataclass
class RegressionOutput:
    """Regression prediction output."""

    predictions: Any  # shape (B,) or (B, D)


@dataclass
class EmbeddingOutput:
    """Pooled / token-level embeddings used for linear probing."""

    embeddings: Any  # shape (B, D) or (B, L, D)
    pooled: bool = True


@dataclass
class LikelihoodOutput:
    """Per-sequence log-likelihood output used for zero-shot scoring."""

    log_likelihood: Any  # shape (B,)
    num_tokens: Optional[Any] = None  # shape (B,)


@dataclass
class GenerationOutput:
    """Free-form sequence generation output."""

    sequences: Sequence[str]
    sampling_params: Dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Abstract adapter exposing a uniform evaluation API.

    Concrete subclasses implement whichever subset of methods the
    underlying architecture supports.  Callers should check
    :meth:`supports` before dispatching.
    """

    handle: ModelHandle

    def __init__(self, handle: ModelHandle):
        self.handle = handle

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer into memory."""

    def close(self) -> None:  # noqa: B027 - intentional no-op default
        """Free model resources.  Default is a no-op."""
        return None

    def supports(self, capability: str) -> bool:
        """Return ``True`` when the adapter implements ``capability``.

        Capabilities used across evaluators:

        * ``"classification"`` - :meth:`predict_classification`
        * ``"regression"`` - :meth:`predict_regression`
        * ``"embedding"`` - :meth:`embed`
        * ``"likelihood"`` - :meth:`score_likelihood`
        * ``"generation"`` - :meth:`generate`
        """
        method = getattr(self, _CAPABILITY_METHOD[capability], None)
        if method is None:
            return False
        # A subclass opts in by overriding the method.
        return method.__func__ is not getattr(ModelAdapter, _CAPABILITY_METHOD[capability])

    # ---- Optional operations (subclasses override the ones they support) ----

    def predict_classification(
        self, inputs: Sequence[str], **kwargs: Any
    ) -> ClassificationOutput:
        raise NotImplementedError(
            f"{type(self).__name__} does not support classification"
        )

    def predict_regression(
        self, inputs: Sequence[str], **kwargs: Any
    ) -> RegressionOutput:
        raise NotImplementedError(
            f"{type(self).__name__} does not support regression"
        )

    def embed(self, inputs: Sequence[str], **kwargs: Any) -> EmbeddingOutput:
        raise NotImplementedError(
            f"{type(self).__name__} does not support embedding"
        )

    def score_likelihood(
        self, inputs: Sequence[str], **kwargs: Any
    ) -> LikelihoodOutput:
        raise NotImplementedError(
            f"{type(self).__name__} does not support likelihood scoring"
        )

    def generate(
        self,
        prompts: Optional[Sequence[str]] = None,
        num_samples: int = 1,
        **kwargs: Any,
    ) -> GenerationOutput:
        raise NotImplementedError(
            f"{type(self).__name__} does not support generation"
        )


_CAPABILITY_METHOD = {
    "classification": "predict_classification",
    "regression": "predict_regression",
    "embedding": "embed",
    "likelihood": "score_likelihood",
    "generation": "generate",
}


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTER_FACTORIES: Dict[str, Callable[[ModelHandle], ModelAdapter]] = {}


def register_adapter(
    arch: str, factory: Callable[[ModelHandle], ModelAdapter]
) -> None:
    """Register ``factory`` as the adapter builder for ``arch``."""
    _ADAPTER_FACTORIES[arch] = factory


def available_adapters() -> List[str]:
    """Return the list of registered architecture tags."""
    return sorted(_ADAPTER_FACTORIES)


def build_adapter(handle: ModelHandle) -> ModelAdapter:
    """Instantiate the adapter registered for ``handle.arch``.

    Raises:
        KeyError: when no adapter is registered for the architecture.
    """
    try:
        factory = _ADAPTER_FACTORIES[handle.arch]
    except KeyError as exc:
        raise KeyError(
            f"No ModelAdapter registered for arch={handle.arch!r}. "
            f"Registered: {available_adapters()}"
        ) from exc
    return factory(handle)


def iter_registered() -> Iterable[str]:
    """Iterate over registered architecture tags (stable order)."""
    return iter(available_adapters())
