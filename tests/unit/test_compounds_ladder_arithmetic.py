"""The step-to-epoch and step-to-token arithmetic the compounds ladder is written from.

These two conversions appear in the ladder table, in the corpus-size table and in
every report that states how long a run was. They are the kind of derived number
that gets copied once and then carried forward, so they are pinned here against
values measured elsewhere rather than recomputed by hand each time.

The reference points:
  - 15,000 steps is 96.26 epochs and 39,321,600,000 tokens (BERT, measured)
  - 30,000 steps is 192.52 epochs
  - 4,674 steps is the GPT-2 schedule, close to 30 epochs by construction
"""

import importlib.util
from pathlib import Path

import pytest

# scripts/ is not a package, so the module is loaded by path, as the clinvar table
# test does.
_SPEC = importlib.util.spec_from_file_location(
    "_compounds_ladder_table",
    Path(__file__).resolve().parents[2] / "scripts" / "compounds_ladder_table.py",
)
_LADDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LADDER)

SEQ_PER_STEP = _LADDER.SEQ_PER_STEP
TOKENS_PER_SEQ = _LADDER.TOKENS_PER_SEQ
TRAIN_WINDOWS = _LADDER.TRAIN_WINDOWS
epochs = _LADDER.epochs
gpt2_param_count = _LADDER.gpt2_param_count
tokens = _LADDER.tokens


def test_the_corpus_constants_are_the_packed_build():
    """398,917 training windows of 1,024 tokens, at a global batch of 2,560."""
    assert TRAIN_WINDOWS == 398_917
    assert SEQ_PER_STEP == 2_560
    assert TOKENS_PER_SEQ == 1_024
    # The whole training split, as reported in the corpus-size table.
    assert TRAIN_WINDOWS * TOKENS_PER_SEQ == 408_491_008


@pytest.mark.parametrize(
    "steps,expected",
    [(15_000, 96.26), (30_000, 192.52), (45_000, 288.78), (4_674, 29.99)],
)
def test_epochs(steps, expected):
    assert round(epochs(steps), 2) == expected


@pytest.mark.parametrize(
    "steps,expected",
    [(15_000, 39_321_600_000), (30_000, 78_643_200_000), (4_674, 12_252_610_560)],
)
def test_tokens(steps, expected):
    assert tokens(steps) == expected


def test_one_step_is_the_tokens_per_step_in_the_corpus_table():
    assert tokens(1) == 2_621_440


def test_the_gpt2_schedule_is_thirty_epochs_rounded_down():
    """4,674 is not an arbitrary number: it is 30 epochs at this batch, truncated."""
    assert int(30 * TRAIN_WINDOWS / SEQ_PER_STEP) == 4_674


def test_gpt2_param_count_against_the_measured_bert_of_the_same_width():
    """A sanity bound, not an equality: the two architectures differ at the head.

    BERT small measures 86,909,800 parameters from its safetensors headers at the
    same hidden size, layer count and vocabulary. The nanoGPT count is derived from
    declared dimensions, so it is checked for being the same order and below BERT's,
    which carries an MLM head BERT needs and nanoGPT does not.
    """
    count = gpt2_param_count(
        {"n_embd": 768, "n_layer": 12, "vocab_size": 612, "n_positions": 1024}
    )
    assert count == 86_210_304
    assert count < 86_909_800
