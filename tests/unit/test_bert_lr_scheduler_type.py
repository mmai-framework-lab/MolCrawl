"""The learning-rate schedule is selectable, and unset means exactly what it meant before.

HF's default schedule is "linear": the rate decays from the peak to exactly 0.0 at
max_steps, with no separate decay length. A run extended past its original max_steps
therefore cannot keep that schedule - the curve is recomputed over the new max_steps
and the continuation is a different run. Selecting a constant schedule is what makes
an extension a continuation.

Declaring the key must not move anything for the configs that do not mention it, so
these tests pin both halves: the default, and that nothing is passed when it is unset.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "molcrawl" / "models" / "bert" / "main.py"


def _assigned_value(name):
    """The value main.py's config block assigns to `name`."""
    for node in ast.walk(ast.parse(MAIN.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in main.py")


def test_unset_by_default_so_hf_keeps_choosing():
    assert _assigned_value("lr_scheduler_type") is None


def test_declared_so_the_configurator_accepts_it_from_the_command_line():
    """configurator.py raises ValueError for a --key it does not find in globals()."""
    src = MAIN.read_text(encoding="utf-8")
    assert re.search(r"^\s*lr_scheduler_type\s*=", src, re.M)


def test_passed_only_when_asked_for():
    """An unset config must produce no lr_scheduler_type key at all.

    Passing it unconditionally would hand TrainingArguments None, which is not a
    SchedulerType, so every existing config would stop building.
    """
    src = MAIN.read_text(encoding="utf-8")
    passthrough = [
        ln for ln in src.splitlines()
        if "lr_scheduler_type" in ln and not ln.lstrip().startswith("#")
    ]
    # the declaration plus the two lines of the conditional spread
    assert len(passthrough) == 3, passthrough
    joined = " ".join(passthrough)
    assert 'globals().get("lr_scheduler_type")' in joined
    assert 'if globals().get("lr_scheduler_type") else {}' in joined


def test_the_documented_values_are_ones_hf_accepts():
    """The comment names "constant" and "constant_with_warmup"; check they are real."""
    from transformers.trainer_utils import SchedulerType

    names = {s.value for s in SchedulerType}
    assert "constant" in names
    assert "constant_with_warmup" in names
    assert "linear" in names  # the default the docstring above claims


def test_the_default_schedule_is_the_one_the_comment_names(tmp_path):
    from transformers import TrainingArguments

    args = TrainingArguments(output_dir=str(tmp_path))
    assert str(args.lr_scheduler_type) in ("SchedulerType.LINEAR", "linear")
