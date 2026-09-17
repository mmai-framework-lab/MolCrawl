"""Every existing config resolves to the batch it already ran with.

The DeepSpeed order (§3.2) makes this the gate for touching the accumulation
rule: the 48 ``gpt2*.py`` configs hold the *total* accumulation, and a resolver
that read them as per-rank would quadruple the batch on 4 GPUs. The protein
GPT-2 retrain is mid-campaign, so a silent change here would split one ladder
into two batch meanings.

So this does not trust the algebra in ``_batch_policy``. For each config and each
world size it runs both computations side by side:

legacy
    what ``gpt2/train.py:344-345`` does today -- assert the accumulation divides
    by the world size, integer-divide, multiply back;
canonical
    ``from_legacy_gpt2`` then ``resolve_global_fixed``;

and requires the same verdict (both run or both refuse) and, when both run, the
same per-rank accumulation and effective global batch.

Configs are read with ``ast``, not executed: they import tokenizers and read
datasets at module level. Only module-level assignments of integer arithmetic are
evaluated; a value this cannot evaluate fails the test rather than being skipped,
so a config written some new way is noticed.
"""

import ast
import glob
import os

import pytest

from molcrawl.models._batch_policy import (
    BatchPolicyError,
    from_legacy_bert,
    from_legacy_gpt2,
    resolve_global_fixed,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS = os.path.join(REPO, "molcrawl", "tasks", "pretrain", "configs")
WORLD_SIZES = (1, 2, 4, 8, 12, 16)

GPT2_CONFIGS = sorted(glob.glob(os.path.join(CONFIGS, "*", "gpt2*.py")))
BERT_CONFIGS = sorted(glob.glob(os.path.join(CONFIGS, "*", "bert_*.py")))


class _Unevaluable(Exception):
    pass


def _eval_int(node, env):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise _Unevaluable(node.id)
    if isinstance(node, ast.BinOp):
        left, right = _eval_int(node.left, env), _eval_int(node.right, env)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    raise _Unevaluable(ast.dump(node))


def module_ints(path):
    """Module-level names bound to integer arithmetic, last assignment wins."""
    env = {}
    unevaluable = {}
    for stmt in ast.parse(open(path).read()).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name, value = stmt.target.id, stmt.value
        else:
            continue
        try:
            env[name] = _eval_int(value, env)
            unevaluable.pop(name, None)
        except _Unevaluable:
            env.pop(name, None)
            unevaluable[name] = ast.unparse(value)
    return env, unevaluable


def _read(path, *names):
    env, unevaluable = module_ints(path)
    for name in names:
        assert name not in unevaluable, f"{path}: {name} = {unevaluable[name]} is not plain integer arithmetic"
    return env


def legacy_gpt2(batch_size, accum_configured, world_size):
    """gpt2/train.py:344-345 and the effective batch it prints."""
    if accum_configured % world_size != 0:
        return None
    per_rank = accum_configured // world_size
    return per_rank, batch_size * per_rank * world_size


def canonical(canon, world_size):
    try:
        r = resolve_global_fixed(canon, world_size)
    except BatchPolicyError:
        return None
    return r.gradient_accumulation_steps_per_rank, r.effective_global_batch


def test_the_gpt2_config_count_is_what_the_order_names():
    assert len(GPT2_CONFIGS) == 48, [os.path.relpath(p, CONFIGS) for p in GPT2_CONFIGS]


@pytest.mark.parametrize("path", GPT2_CONFIGS, ids=lambda p: os.path.relpath(p, CONFIGS))
def test_gpt2_config_resolves_to_the_batch_it_ran_with(path):
    env = _read(path, "batch_size", "gradient_accumulation_steps")
    assert "batch_size" in env and "gradient_accumulation_steps" in env, path
    bs, accum = env["batch_size"], env["gradient_accumulation_steps"]
    canon = from_legacy_gpt2(bs, accum)
    for world in WORLD_SIZES:
        assert canonical(canon, world) == legacy_gpt2(bs, accum, world), (path, world)
    # The declared guard, where there is one, agrees with the canonical target.
    if "expected_global_batch" in env:
        assert canon.target_global_batch == env["expected_global_batch"], path


def test_gpt2_world_sizes_that_legacy_refuses_are_the_ones_canonical_refuses():
    """The counts the handoff quotes: all 48 fail at 12 GPUs, the accum-40 ones at 16."""
    refused = {w: 0 for w in WORLD_SIZES}
    for path in GPT2_CONFIGS:
        env = _read(path, "batch_size", "gradient_accumulation_steps")
        canon = from_legacy_gpt2(env["batch_size"], env["gradient_accumulation_steps"])
        for w in WORLD_SIZES:
            refused[w] += canonical(canon, w) is None
    assert refused == {1: 0, 2: 0, 4: 0, 8: 0, 12: 48, 16: 18}


def _bert_declared():
    out = []
    for path in BERT_CONFIGS:
        env, unevaluable = module_ints(path)
        if "expected_global_batch" in env or "expected_global_batch" in unevaluable:
            out.append(path)
    return out


@pytest.mark.parametrize("path", _bert_declared(), ids=lambda p: os.path.relpath(p, CONFIGS))
def test_declared_bert_config_resolves_at_four_gpus_to_its_own_accumulation(path):
    """HF reads accum per rank; the declaration was written for 4 GPUs."""
    env = _read(path, "batch_size", "gradient_accumulation_steps", "expected_global_batch")
    canon = from_legacy_bert(env["batch_size"], env["gradient_accumulation_steps"], env["expected_global_batch"])
    r = resolve_global_fixed(canon, 4)
    assert r.gradient_accumulation_steps_per_rank == env["gradient_accumulation_steps"], path
    # HF's own arithmetic at the same world size gives the same batch.
    assert env["batch_size"] * env["gradient_accumulation_steps"] * 4 == r.effective_global_batch


def test_undeclared_bert_configs_are_refused():
    declared = set(_bert_declared())
    undeclared = [p for p in BERT_CONFIGS if p not in declared]
    assert undeclared, "every BERT config declares expected_global_batch; update this test"
    for path in undeclared:
        env, _ = module_ints(path)
        with pytest.raises(BatchPolicyError):
            from_legacy_bert(env.get("batch_size", 1), env.get("gradient_accumulation_steps", 1), None)
