"""Ambiguity-aware data collators for MLM and CLM pretraining.

Loss handling for IUPAC ambiguity codes is modality-specific (e.g. genome's
``N`` and DNA ambiguity codes are not predictable single residues, while
protein's ``U``/``O`` are real amino acids that should be predicted). This
module supplies the policy plumbing:

- ``AmbiguityAwareMLMCollator``: wraps HuggingFace's
  ``DataCollatorForLanguageModeling`` and post-processes its output so that
  positions whose *original* input id is in ``ambiguous_tokens`` are excluded
  from loss (labels set to the model's ignore_index, default ``-1``).
- ``mask_ambiguous_targets_for_clm``: small helper for custom-loop CLM trainers
  (nanoGPT-style). Apply to the *shifted* target ``y = batch[:, 1:]`` so the
  loss skips positions where the prediction target is ambiguous.

Implementation choice: the MLM side uses a **wrapper** (composition) rather
than subclassing ``DataCollatorForLanguageModeling`` and overriding its
internal ``torch_mask_tokens``. That internal method has been renamed across
HF versions in the past; wrapping the public ``__call__`` contract is far less
brittle, at the cost of letting the base collator pick ambiguous positions as
mask candidates (whose contribution we then strip in post). The "wasted mask
budget" is at most a few percent in realistic corpora and has no correctness
impact — the loss still ignores them.

Sentinel value: ``-1`` (not the HF-standard ``-100``) to stay aligned with
``molcrawl/models/{gpt2,llama}/model.py`` which compute cross-entropy with
``ignore_index=-1``. If the model side is ever migrated to ``-100``, only the
``IGNORE_INDEX`` constant below needs to change.

See ``docs/_tmp/20260516-spec02-ambiguity-review.md`` for the design review
and policy rationale.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

try:
    from transformers import DataCollatorForLanguageModeling
except ImportError:  # pragma: no cover - documentation-only environment
    DataCollatorForLanguageModeling = None  # type: ignore[assignment]


# The ignore_index used by molcrawl/models/{gpt2,llama}/model.py (cross_entropy).
# Keep this in sync if the model side ever migrates to HF-standard -100.
IGNORE_INDEX: int = -1


# Convenience constants for modality-specific ambiguous tokens.
GENOME_AMBIGUOUS_TOKENS: List[str] = [
    "N",                                              # any base
    "R", "Y", "K", "M", "S", "W",                     # 2-way IUPAC
    "B", "D", "H", "V",                               # 3-way IUPAC
]

# Protein ambiguity / structural markers. U (Sec) and O (Pyl) are deliberately
# NOT listed: they are real amino acids (21st and 22nd) and should be predicted
# normally. The structural markers `.`, `-`, `|` are present in MolCrawl's
# protein SEQUENCE_VOCAB (see molcrawl/data/protein_sequence/dataset/tokenizer.py)
# and `|` is used as the chain-break token.
PROTEIN_AMBIGUOUS_TOKENS: List[str] = [
    "X", "B", "Z",       # truly ambiguous amino acids
    ".", "-", "|",       # structural / chain-break markers
]


# =========================================================================
# Tokenizer-agnostic helpers
# =========================================================================
def _token_to_id(tokenizer, token: str) -> Optional[int]:
    """Resolve a token string to its id, supporting HF tokenizers and SentencePiece."""
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        tid = tokenizer.convert_tokens_to_ids(token)
        return tid
    if hasattr(tokenizer, "piece_to_id"):
        return tokenizer.piece_to_id(token)
    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)!r}")


def _unk_token_id(tokenizer) -> Optional[int]:
    if hasattr(tokenizer, "unk_token_id"):
        return tokenizer.unk_token_id
    if hasattr(tokenizer, "unk_id"):
        return tokenizer.unk_id()
    return None


def resolve_ambiguous_token_ids(
    tokenizer,
    ambiguous_tokens: Sequence[str],
    *,
    log: bool = True,
) -> List[int]:
    """Resolve a list of token strings to ids, skipping any that map to unk.

    Set ``log=True`` (default) to print which symbols were resolved vs.
    skipped — important visibility, because a typo or a vocab mismatch
    otherwise silently disables ambiguity handling.
    """
    unk = _unk_token_id(tokenizer)
    resolved: List[int] = []
    missing: List[str] = []
    for tok in ambiguous_tokens:
        tid = _token_to_id(tokenizer, tok)
        if tid is None or (unk is not None and tid == unk):
            missing.append(tok)
            continue
        resolved.append(int(tid))
    if log:
        print(
            f"[ambiguity] resolved {len(resolved)}/{len(ambiguous_tokens)} tokens "
            f"to ids; ignore_index={IGNORE_INDEX}; missing={missing}"
        )
    return resolved


# =========================================================================
# MLM (encoder-side) collator
# =========================================================================
class AmbiguityAwareMLMCollator:
    """Wrapper around DataCollatorForLanguageModeling that zeroes loss
    on positions whose *original* input id is ambiguous.

    Wrapper (composition) is chosen over subclassing because the base
    collator's internal ``torch_mask_tokens`` API has changed across
    HF transformers versions. The ``__call__`` contract is stable.
    """

    def __init__(
        self,
        tokenizer,
        ambiguous_tokens: Sequence[str],
        *,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        **base_kwargs: Any,
    ) -> None:
        if DataCollatorForLanguageModeling is None:
            raise ImportError(
                "transformers is required for AmbiguityAwareMLMCollator"
            )
        self._base = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            **base_kwargs,
        )
        self._ambiguous_ids = resolve_ambiguous_token_ids(tokenizer, ambiguous_tokens)
        # Expose tokenizer so callers (e.g. HF Trainer) can introspect.
        self.tokenizer = tokenizer

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        batch = self._base(examples)
        if not self._ambiguous_ids:
            return batch

        input_ids: torch.Tensor = batch["input_ids"]
        labels: torch.Tensor = batch["labels"]

        # After the base collator runs, `labels` holds the ORIGINAL token id
        # at masked positions (and IGNORE for non-masked positions). So a
        # position whose original input was ambiguous can be detected by
        # checking labels (not input_ids, which may have been replaced with
        # [MASK] / random / kept depending on the 80/10/10 split).
        ambig_mask = torch.zeros_like(labels, dtype=torch.bool)
        for tok_id in self._ambiguous_ids:
            ambig_mask |= labels == tok_id
            # Cover the 10%-keep case (label retains original id but input is
            # the same) and the non-masked case (label is IGNORE already).
            # input_ids hits cover positions retained or unmasked but with
            # ambiguous input — only matters when the base collator's default
            # IGNORE differs from IGNORE_INDEX.
            ambig_mask |= input_ids == tok_id

        labels[ambig_mask] = IGNORE_INDEX
        batch["labels"] = labels
        return batch


def make_mlm_collator(
    tokenizer,
    ambiguous_tokens: Sequence[str],
    *,
    mlm_probability: float = 0.15,
    **base_kwargs: Any,
):
    """Factory: return a plain ``DataCollatorForLanguageModeling`` when
    ``ambiguous_tokens`` is empty (no overhead), otherwise an
    ``AmbiguityAwareMLMCollator``.
    """
    if DataCollatorForLanguageModeling is None:
        raise ImportError("transformers is required for make_mlm_collator")
    if not ambiguous_tokens:
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, **base_kwargs
        )
    return AmbiguityAwareMLMCollator(
        tokenizer=tokenizer,
        ambiguous_tokens=ambiguous_tokens,
        mlm=True,
        mlm_probability=mlm_probability,
        **base_kwargs,
    )


# =========================================================================
# CLM (decoder-side) helper
# =========================================================================
def mask_ambiguous_targets_for_clm(
    targets: torch.Tensor,
    ambiguous_token_ids: Iterable[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Return a copy of ``targets`` with ambiguous-token positions set to
    ``ignore_index``.

    Apply to the *shifted* target ``y = batch[:, 1:]``, NOT to the input
    ``x = batch[:, :-1]``: it is the predicted (next) token whose loss we
    want to silence at ambiguous positions.
    """
    ambiguous_token_ids = list(ambiguous_token_ids)
    if not ambiguous_token_ids:
        return targets
    out = targets.clone()
    for tok_id in ambiguous_token_ids:
        out[out == tok_id] = ignore_index
    return out


# =========================================================================
# Modality dispatch
# =========================================================================
MODALITY_TO_AMBIGUOUS: Dict[str, List[str]] = {
    "genome_sequence": GENOME_AMBIGUOUS_TOKENS,
    "protein_sequence": PROTEIN_AMBIGUOUS_TOKENS,
    "compounds": [],
    "rna": [],
    "molecule_nat_lang": [],
}


def ambiguous_tokens_for_modality(modality: str) -> List[str]:
    """Look up the ambiguous-token list for a modality string.

    Unknown modality returns an empty list (no-op collator).
    """
    return MODALITY_TO_AMBIGUOUS.get(modality, [])


def infer_modality_from_path(path: Optional[str]) -> Optional[str]:
    """Best-effort modality detection from a dataset/output path.

    Returns the first modality name whose token appears in the path, or None
    if no known modality matches. Used by main.py / train.py scripts that
    don't expose ``modality`` as a config variable but do know
    ``dataset_dir`` or ``out_dir``.
    """
    if not path:
        return None
    p = str(path)
    # Order matters: longer / more-specific names first.
    for modality in ("genome_sequence", "protein_sequence", "molecule_nat_lang", "compounds", "rna"):
        if modality in p:
            return modality
    return None
