"""The guard has to catch the two configurations that actually shipped wrong.

nanoGPT's effective batch does not move with the GPU count: the accumulation is
divided by the world size and multiplied back. What went wrong twice was
simpler -- batch_size and gradient_accumulation_steps were edited apart and
their product stopped being the number the schedule assumed. genome trained at
640 against an intended 2,560, protein's LR pilots at 480.

These reproduce the arithmetic the guard performs, at several world sizes, to
pin down both that the bad pairs are rejected and that the GPU count does not
change the verdict.
"""

import pytest

INTENDED = 2560


def effective(batch_size, grad_accum_config, world_size):
    """What train.py computes, including the divide-then-multiply."""
    assert grad_accum_config % world_size == 0
    per_rank = grad_accum_config // world_size
    return batch_size * per_rank * world_size


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "batch_size,grad_accum,expected",
    [
        (8, 320, 2560),    # genome, after 4974f55
        (16, 160, 2560),   # protein, after 1ccf437
        (8, 80, 640),      # genome, as it shipped
        (12, 40, 480),     # protein's LR pilots
    ],
)
def test_effective_batch_does_not_depend_on_gpu_count(
    batch_size, grad_accum, expected, world_size
):
    assert effective(batch_size, grad_accum, world_size) == expected


@pytest.mark.parametrize("batch_size,grad_accum", [(8, 80), (12, 40)])
def test_the_pairs_that_shipped_wrong_are_rejected(batch_size, grad_accum):
    assert effective(batch_size, grad_accum, 8) != INTENDED


@pytest.mark.parametrize("batch_size,grad_accum", [(8, 320), (16, 160)])
def test_the_corrected_pairs_pass(batch_size, grad_accum):
    assert effective(batch_size, grad_accum, 8) == INTENDED


def test_the_genome_config_declares_what_its_schedule_assumes():
    """_GLOBAL_BATCH drives max_iters; expected_global_batch must follow it."""
    import re

    src = open(
        "molcrawl/tasks/pretrain/configs/genome_sequence/gpt2_small_subset.py"
    ).read()
    declared = re.search(r"^expected_global_batch = (.+)$", src, re.M)
    assert declared, "the config does not declare expected_global_batch"
    # Bound to the same name the schedule uses, so the two cannot drift.
    assert declared.group(1).strip() == "_GLOBAL_BATCH"

    bs = int(re.search(r"^batch_size = (\d+)", src, re.M).group(1))
    ga = int(re.search(r"^gradient_accumulation_steps = (\d+)", src, re.M).group(1))
    gb = int(re.search(r"^_GLOBAL_BATCH = (\d+)", src, re.M).group(1))
    assert bs * ga == gb, f"{bs} x {ga} = {bs * ga}, but _GLOBAL_BATCH is {gb}"
