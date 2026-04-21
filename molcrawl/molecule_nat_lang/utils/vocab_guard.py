"""Fail-fast vocab_size guard for molecule_nat_lang configs.

All molecule_nat_lang pretraining and fine-tuning checkpoints produced so far
were trained with the GPT-2 tokenizer (vocab_size=50257). BERT-based configs
round the vocabulary up to the next multiple of 8 (50264) for efficient
embedding lookups; the GPT-2 configs use the raw size (50257). If the
tokenizer is ever swapped (e.g. to CodeLlama, or to a local GPT-2 copy with
extra special tokens), the derived ``meta_vocab_size`` will differ from the
expected value and the resulting model will not be compatible with the
existing checkpoints.

Past experience: silent vocab drift (BasicTokenizer fallback → 50002, or
CodeLlama → 32024) produced checkpoints that looked fine but actually had
random decoder weights, wasting days of compute. Call
``check_vocab_size(meta_vocab_size, expected=EXPECTED_VOCAB_SIZE_BERT)`` (or
``EXPECTED_VOCAB_SIZE_GPT2``) at the bottom of every molecule_nat_lang config
so the mismatch crashes loudly at startup instead.

If you intentionally want to retrain with a different tokenizer, update the
constants here — the change will be visible in a single diff.
"""

EXPECTED_VOCAB_SIZE_GPT2 = 50257  # GPT-2 tokenizer, raw (used by nanoGPT-style configs)
EXPECTED_VOCAB_SIZE_BERT = 50264  # GPT-2 padded to a multiple of 8 (used by HF BERT configs)

# Backwards-compatible default for callers that don't specify — BERT is the
# more common case in this codebase.
EXPECTED_VOCAB_SIZE = EXPECTED_VOCAB_SIZE_BERT


def check_vocab_size(meta_vocab_size: int, *, expected: int = EXPECTED_VOCAB_SIZE) -> None:
    """Raise ``RuntimeError`` unless ``meta_vocab_size == expected``.

    Intended to be called once per config at module load, after
    ``meta_vocab_size`` has been derived from the tokenizer. BERT configs
    should pass ``expected=EXPECTED_VOCAB_SIZE_BERT`` (50264, padded) and
    GPT-2 configs ``expected=EXPECTED_VOCAB_SIZE_GPT2`` (50257, raw).
    """
    if meta_vocab_size == expected:
        return

    raise RuntimeError(
        f"molecule_nat_lang meta_vocab_size mismatch: expected {expected} "
        f"(GPT-2 tokenizer, {'padded to multiple of 8' if expected == EXPECTED_VOCAB_SIZE_BERT else 'raw'}), "
        f"got {meta_vocab_size}.\n"
        "Existing pretraining checkpoints were trained with the GPT-2 "
        "tokenizer. If the active tokenizer has changed (e.g. CodeLlama was "
        "made available, GPT2_TOKENIZER_DIR points to a different tokenizer, "
        "or a silent fallback kicked in) the resulting model will NOT be "
        "compatible with the existing checkpoints.\n"
        "Either restore the GPT-2 tokenizer (populate the HuggingFace cache "
        "with `huggingface-cli download gpt2` and unset any overriding env "
        "vars) or — if this change is intentional — update the expected "
        "constants in molcrawl/molecule_nat_lang/utils/vocab_guard.py AND "
        "retrain the pretrained checkpoints from scratch."
    )
