"""The reasons to write a checkpoint add up; they do not shadow each other.

Chained with ``elif``, setting ``save_checkpoint_steps`` silently dropped every
improvement that did not land on a multiple of the interval -- the best-val
checkpoint was then never written at all, while the run looked normal. GPT-2 was
fixed; llama carried the same lines, and llama_small_extend and llama_xl are
exactly the configuration that triggers it (save_checkpoint_steps set,
always_save_checkpoint off).

Asserted on the source rather than by running a training loop: the expression
sits inside the loop, and what went wrong was its shape.
"""
import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "molcrawl" / "models"
_TRAINERS = {"gpt2": _ROOT / "gpt2" / "train.py", "llama": _ROOT / "llama" / "train.py"}

_REASONS = {"always_save_checkpoint", "save_checkpoint_steps", "is_best_model"}


def _save_decision(path):
    """The single assignment to should_save_checkpoint in the training loop."""
    tree = ast.parse(path.read_text())
    found = [n for n in ast.walk(tree)
             if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == "should_save_checkpoint" for t in n.targets)]
    assert found, f"no should_save_checkpoint assignment in {path}"
    return found


@pytest.mark.parametrize("arch", sorted(_TRAINERS))
def test_the_decision_is_one_expression_not_a_chain(arch):
    assigns = _save_decision(_TRAINERS[arch])

    assert len(assigns) == 1, (
        f"{arch}: {len(assigns)} assignments to should_save_checkpoint. A chain of "
        "them is how an earlier branch comes to shadow a later one."
    )


@pytest.mark.parametrize("arch", sorted(_TRAINERS))
def test_every_reason_is_ored_in(arch):
    node = _save_decision(_TRAINERS[arch])[0].value

    assert isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or), (
        f"{arch}: should_save_checkpoint is not an `or` of its reasons"
    )
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    assert _REASONS <= names, f"{arch}: missing reasons {_REASONS - names}"


@pytest.mark.parametrize("arch", sorted(_TRAINERS))
def test_no_elif_stands_between_the_reasons(arch):
    """The shape of the bug: `if always_save ... elif save_steps ... elif best`."""
    src = _TRAINERS[arch].read_text()
    start = src.index("should_save_checkpoint")
    window = src[max(0, start - 400):start + 400]

    assert "elif save_checkpoint_steps" not in window, (
        f"{arch}: save_checkpoint_steps is chained with elif again"
    )
