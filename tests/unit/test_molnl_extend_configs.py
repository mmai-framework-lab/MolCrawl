"""The seven mol_nl extension configs of the 2026-11-16 order say what they must.

Two things about them cannot be left to a reading of the file. Early stopping is what
decides where these runs end, and the grid configs they import set ``early_stopping =
False``, so the trainer default of True does not reach them: a config that forgets to
state it would run to its ceiling and answer a different question. And the rate past the
original schedule has to be constant -- BERT by ``lr_scheduler_type``, GPT-2 by leaving
``lr_decay_iters`` where it was -- because both frameworks otherwise recompute the whole
curve over the new length and continue a different run.

Everything else about these configs has to be the finished run's, so each arm is also
checked against the config it extends, key by key.

Configs are read with ``ast``, not imported: importing one builds a tokenizer and reads
the data tree. Star-imports are followed, so a value an extension inherits is read with
the number it actually runs with.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / "molcrawl/tasks/pretrain/configs"
MOLNL = CONFIGS / "molecule_nat_lang"

#: The ceiling each extension config carries. The first pass used 48,000 for BERT; the
#: moment-keeping rerun uses 72,000 so that early stopping, not the ceiling, ends the run
#: -- 48,000 did not deliver that for small, which reached it with its minimum at 43,700.
CEILING = {"_ext": {"bert": 48000, "gpt2": 3000}, "_extm": {"bert": 72000}}

#: extension config -> the config it extends
PAIRS = {
    "bert_small_24k_lr1e3_extm": "bert_small_24k_lr1e3",
    "bert_medium_24k_lr3e4_extm": "bert_medium_24k_lr3e4",
    "bert_large_24k_lr1p7e4_extm": "bert_large_24k_lr1p7e4",
    "bert_small_24k_lr1e3_ext": "bert_small_24k_lr1e3",
    "bert_medium_24k_lr3e4_ext": "bert_medium_24k_lr3e4",
    "bert_large_24k_lr1p7e4_ext": "bert_large_24k_lr1p7e4",
    "gpt2_small_1500_lr1p2e3_ext": "gpt2_small_1500_lr1p2e3",
    "gpt2_medium_1500_lr6e4_ext": "gpt2_medium_1500_lr6e4",
    "gpt2_large_1500_lr3e4_ext": "gpt2_large_1500_lr3e4",
    "gpt2_xl_1500_lr3e4_ext": "gpt2_xl_1500_lr3e4",
}

#: Keys an extension must carry over unchanged from the run it continues. The schedule
#: keys are deliberately absent: those are the ones an extension is allowed to move.
SHARED_BERT = ("seed", "model_size", "batch_size", "gradient_accumulation_steps",
               "per_device_eval_batch_size", "expected_global_batch", "max_length",
               "document_masking", "fixed_eval_mask", "eval_mask_seed", "log_interval",
               "save_steps", "save_on_improve", "bf16", "dataloader_num_workers",
               "dataloader_pin_memory", "weight_decay")
SHARED_GPT2 = ("seed", "batch_size", "gradient_accumulation_steps", "block_size",
               "eval_interval", "eval_sequences", "expected_global_batch", "dtype",
               "warmup_iters", "learning_rate", "min_lr", "weight_decay",
               "save_checkpoint_steps", "max_checkpoints", "always_save_checkpoint")

_BINOPS = {ast.Mult: lambda a, b: a * b, ast.Add: lambda a, b: a + b,
           ast.Sub: lambda a, b: a - b, ast.Div: lambda a, b: a / b}


def _value(node, env):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _value(node.operand, env)
        return -inner if isinstance(inner, (int, float)) else None
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _value(node.left, env), _value(node.right, env)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return _BINOPS[type(node.op)](left, right)
    return None


def read(path: Path, seen=None) -> dict:
    """Every literal assignment in the file, with star-imported ones underneath."""
    seen = set() if seen is None else seen
    if path in seen or not path.exists():
        return {}
    seen.add(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            parts = (node.module or "").split(".")
            if "configs" in parts:
                base = CONFIGS.joinpath(*parts[parts.index("configs") + 1:]).with_suffix(".py")
                values.update(read(base, seen))
    env = dict(values)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        got = _value(node.value, env) if node.value is not None else None
        for name in names:
            if got is not None:
                env[name] = got
                values[name] = got
    return values


def test_the_rerun_configs_keep_the_moments_by_asking_for_a_re_pointed_scheduler():
    """The staged copy decides it, and the config is where the rate comes from.

    Removing scheduler.pt fixes the rate and loses the Adam moments, because transformers
    guards both loads with one condition on that file. The rerun arms are staged with the
    file re-pointed at the config instead, so the rate the config declares has to be the
    one the staging writes -- which it reads from here.
    """
    for ext in (name for name in PAIRS if name.endswith("_extm")):
        values = read(MOLNL / f"{ext}.py")
        assert values["lr_scheduler_type"] == "constant_with_warmup", ext
        assert values["warmup_steps"] == 500, ext
        assert values["early_stopping"] is True, ext
        assert values["early_stopping_patience"] == 90, ext


def test_all_seven_extension_configs_exist():
    missing = [name for name in PAIRS if not (MOLNL / f"{name}.py").exists()]
    assert missing == [], missing


def test_every_extension_turns_early_stopping_on_itself():
    """The parents set it to False, so an inherited value is the wrong one."""
    for ext, parent in PAIRS.items():
        assert read(MOLNL / f"{parent}.py")["early_stopping"] is False, parent
        values = read(MOLNL / f"{ext}.py")
        assert values["early_stopping"] is True, ext
        expected = 90 if ext.startswith("bert") else 10
        assert values["early_stopping_patience"] == expected, (ext, values["early_stopping_patience"])


def test_the_bert_extensions_run_at_a_constant_tenth_of_their_peak():
    for ext, parent in PAIRS.items():
        if not ext.startswith("bert"):
            continue
        base, values = read(MOLNL / f"{parent}.py"), read(MOLNL / f"{ext}.py")
        assert values["lr_scheduler_type"] == "constant_with_warmup", ext
        assert values["warmup_steps"] == 500, ext
        # A tenth, compared as a ratio: 3e-5 and 3e-4 / 10 are different floats
        # (2.9999999999999997e-05), and writing the division into the config to make them
        # equal would put arithmetic where the rate a run trains at should be readable.
        assert math.isclose(values["learning_rate"], base["learning_rate"] / 10, rel_tol=1e-9), (
            ext, values["learning_rate"], base["learning_rate"])
        assert base["max_steps"] == 24000, parent
        ceiling = CEILING["_extm" if ext.endswith("_extm") else "_ext"]["bert"]
        assert values["max_steps"] == ceiling, (ext, values["max_steps"], ceiling)


def test_the_gpt2_extensions_leave_the_decay_length_where_it_was():
    """get_lr returns min_lr past lr_decay_iters, which is the whole rate decision."""
    for ext, parent in PAIRS.items():
        if not ext.startswith("gpt2"):
            continue
        base, values = read(MOLNL / f"{parent}.py"), read(MOLNL / f"{ext}.py")
        assert base["lr_decay_iters"] == 1500 and base["max_iters"] == 1500, parent
        assert values["lr_decay_iters"] == 1500, ext
        assert values["max_iters"] == 3000, ext
        assert values["init_from"] == "resume", ext
        assert math.isclose(values["min_lr"], values["learning_rate"] / 10, rel_tol=1e-9), ext


def test_nothing_but_the_schedule_moves():
    for ext, parent in PAIRS.items():
        base, values = read(MOLNL / f"{parent}.py"), read(MOLNL / f"{ext}.py")
        shared = SHARED_BERT if ext.startswith("bert") else SHARED_GPT2
        for key in shared:
            assert key in base, (parent, key)
            assert values.get(key) == base[key], (ext, key, values.get(key), base[key])


def test_no_extension_config_declares_a_seed_of_its_own():
    """Seed 42 comes from the arm being continued; a second declaration could drift."""
    for ext in PAIRS:
        own = ast.parse((MOLNL / f"{ext}.py").read_text(encoding="utf-8"))
        names = [t.id for node in own.body if isinstance(node, ast.Assign)
                 for t in node.targets if isinstance(t, ast.Name)]
        assert "seed" not in names, ext
        assert read(MOLNL / f"{ext}.py")["seed"] == 42, ext
