"""The batch policy resolver: arithmetic, refusals, and the scaling table.

The feasibility table is the one in the DeepSpeed order (§12.1): target 2,560 on
4-GPU nodes, micro batch 8 / 32 / 160 over 1, 2, 4 and 8 nodes.
"""

import pytest

from molcrawl.models._batch_policy import (
    GLOBAL_FIXED,
    PER_GPU_FIXED,
    BatchPolicyError,
    CanonicalBatch,
    from_legacy_bert,
    from_legacy_gpt2,
    resolve,
    resolve_global_fixed,
    resolve_per_gpu_fixed,
    scaling_feasibility,
)


def _canon(micro, target):
    return CanonicalBatch(micro_batch_size_per_gpu=micro, target_global_batch=target, source="test")


@pytest.mark.parametrize("world_size,per_rank", [(1, 320), (2, 160), (4, 80), (8, 40), (16, 20)])
def test_global_fixed_derives_accumulation_and_holds_the_batch(world_size, per_rank):
    r = resolve_global_fixed(_canon(8, 2560), world_size)
    assert r.gradient_accumulation_steps_per_rank == per_rank
    assert r.effective_global_batch == 2560
    assert r.policy == GLOBAL_FIXED


@pytest.mark.parametrize("micro,world_size", [(32, 32), (160, 32), (8, 12), (7, 4)])
def test_global_fixed_refuses_instead_of_rounding(micro, world_size):
    with pytest.raises(BatchPolicyError, match="not a positive integer"):
        resolve_global_fixed(_canon(micro, 2560), world_size)


@pytest.mark.parametrize("world_size", [1, 4, 8, 16])
def test_per_gpu_fixed_grows_the_global_batch_with_world_size(world_size):
    r = resolve_per_gpu_fixed(32, 5, world_size)
    assert r.gradient_accumulation_steps_per_rank == 5
    assert r.effective_global_batch == 32 * 5 * world_size
    assert r.target_global_batch is None
    assert r.policy == PER_GPU_FIXED


def test_unknown_policy_is_refused():
    with pytest.raises(BatchPolicyError, match="unknown batch_policy"):
        resolve("per_node_fixed", 4, canonical=_canon(8, 2560))


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "8", None])
def test_non_positive_or_non_int_inputs_are_refused(bad):
    with pytest.raises(BatchPolicyError):
        resolve_global_fixed(_canon(8, 2560), bad)


def test_legacy_gpt2_target_is_the_configured_product():
    c = from_legacy_gpt2(16, 160)
    assert (c.micro_batch_size_per_gpu, c.target_global_batch) == (16, 2560)
    assert c.legacy == {"batch_size": 16, "gradient_accumulation_steps_configured": 160}


def test_legacy_bert_uses_the_declared_global_batch():
    c = from_legacy_bert(8, 80, 2560)
    assert (c.micro_batch_size_per_gpu, c.target_global_batch) == (8, 2560)
    assert resolve_global_fixed(c, 4).gradient_accumulation_steps_per_rank == 80


def test_legacy_bert_without_a_declaration_is_refused_not_guessed():
    with pytest.raises(BatchPolicyError, match="does not declare expected_global_batch"):
        from_legacy_bert(8, 80, None)


def test_scaling_feasibility_matches_the_order_table():
    expected = {
        8: {1: 80, 2: 40, 4: 20, 8: 10},
        32: {1: 20, 2: 10, 4: 5, 8: None},
        160: {1: 4, 2: 2, 4: 1, 8: None},
    }
    for micro, by_nodes in expected.items():
        table = scaling_feasibility(_canon(micro, 2560))
        got = {c["nodes"]: c["gradient_accumulation_steps_per_rank"] for c in table["candidates"]}
        assert got == by_nodes, micro
        assert table["max_feasible_candidate_node_count"] == max(n for n, v in by_nodes.items() if v)
        for c in table["candidates"]:
            assert c["world_size"] == c["nodes"] * 4
            assert c["feasible"] is (c["reason"] is None)


def test_scaling_feasibility_says_why_a_candidate_fails():
    table = scaling_feasibility(_canon(160, 2560))
    eight = [c for c in table["candidates"] if c["nodes"] == 8][0]
    assert eight["quotient"] == 0.5
    assert "< 1" in eight["reason"]
    table = scaling_feasibility(_canon(32, 2560))
    eight = [c for c in table["candidates"] if c["nodes"] == 8][0]
    assert eight["quotient"] == 2.5
    assert "not an integer" in eight["reason"]


def test_manifest_view_carries_formula_and_tokens():
    r = resolve_global_fixed(from_legacy_gpt2(8, 320), 4)
    m = r.as_manifest(sequence_length=1024)
    assert m["formula"] == "2560 / (8 x 4) = 80"
    assert m["global_tokens_per_optimizer_step"] == 2560 * 1024
    assert m["legacy"]["gradient_accumulation_steps_configured"] == 320
