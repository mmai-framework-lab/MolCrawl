"""eval_iters is derived from eval_sequences, so a ladder compares equal samples.

eval_iters counts eval *batches* and the ladders shrink batch_size as the model
grows, so a shared eval_iters silently gives the larger models a smaller validation
sample. These cover the arithmetic and the config wiring that replaces it.
"""
import pathlib

import pytest

from molcrawl.models.gpt2.train import resolve_eval_iters

CONFIG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "molcrawl/tasks/pretrain/configs"
LADDERS = ["compounds", "protein_sequence", "rna", "molecule_nat_lang"]
SIZES = ["small", "medium", "large", "xl"]


@pytest.mark.parametrize(
    "sequences,batch,expected",
    [
        (3200, 16, 200),   # small / medium
        (3200, 8, 400),    # large
        (3200, 4, 800),    # xl
        (3200, 12, 267),   # rounds up rather than under-sampling
        (10, 16, 1),       # never zero batches
    ],
)
def test_resolve_eval_iters(sequences, batch, expected):
    assert resolve_eval_iters(sequences, batch) == expected


def test_every_ladder_size_covers_at_least_the_requested_sequences():
    for modality in LADDERS:
        for size in SIZES:
            text = (CONFIG_ROOT / modality / f"gpt2_{size}.py").read_text(encoding="utf-8")
            values = dict(
                line.split(" = ", 1) for line in text.splitlines()
                if line.startswith(("eval_sequences = ", "batch_size = "))
            )
            sequences = int(values["eval_sequences"])
            batch = int(values["batch_size"].split("#")[0])
            assert resolve_eval_iters(sequences, batch) * batch >= sequences


def test_ladder_configs_agree_on_the_sample_size():
    """A size-dependent eval_sequences would reintroduce exactly what this removes."""
    for modality in LADDERS:
        seen = set()
        for size in SIZES:
            text = (CONFIG_ROOT / modality / f"gpt2_{size}.py").read_text(encoding="utf-8")
            seen.update(
                line.split(" = ", 1)[1] for line in text.splitlines()
                if line.startswith("eval_sequences = ")
            )
        assert seen == {"3200"}, f"{modality}: {seen}"


def test_ladder_configs_do_not_also_set_eval_iters():
    """train.py derives eval_iters, so setting it here would read as live but be dead."""
    for modality in LADDERS:
        for size in SIZES:
            path = CONFIG_ROOT / modality / f"gpt2_{size}.py"
            lines = path.read_text(encoding="utf-8").splitlines()
            assert not [line for line in lines if line.startswith("eval_iters")], path
