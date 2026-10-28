"""The four compounds extensions to 45,000 steps continue the 30,000-step runs unchanged.

The 2026-10-28 directive says all four sizes go to 45,000 (§2) and that the
continuation runs at the same tenth of the original peak as the 30,000-step
extension, without being changed part-way (§3). A transcription slip in a rate
would not fail anything at launch: the run would simply train at the wrong rate
and report a number under a condition nobody chose. So every value is compared
against the ext30k config it continues, and only the three that are meant to
differ are allowed to.

The separate output directory matters for the same reason the ext30k one did.
Retention keeps the best ten checkpoints, so writing into the 30,000-step
directory would prune the checkpoints the ladder values reported on 2026-10-27
point at.
"""

import ast
from pathlib import Path

import pytest

CONFIG_DIR = (
    Path(__file__).resolve().parents[2]
    / "molcrawl"
    / "tasks"
    / "pretrain"
    / "configs"
    / "compounds"
)

# size : the stem shared by the ext30k config and its ext45k continuation
ARMS = {
    "small": "bert_small_lr1e3_seed42",
    "medium": "bert_medium_lr2p8e4_seed42",
    "large": "bert_large_lr1p5e4_seed42",
    "xlarge": "bert_xl_lr4p5e5_seed42",
}

# The keys the continuation is meant to change. max_steps is the point of it;
# model_path keeps the 30,000-step directory intact.
EXPECTED_DIFFERENCES = {"max_steps", "model_path"}


def assignments(path):
    """Every module-level ``name = literal`` in a config, as a dict.

    Configs are read rather than imported: importing one pulls in the tokenizer
    and the dataset paths, which a unit test has no business touching.
    """
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            try:
                out[target.id] = ast.literal_eval(node.value)
            except ValueError:
                # model_path is a call on get_bert_output_path; keep the suffix,
                # which is the part that decides the directory.
                out[target.id] = ast.dump(node.value)
    return out


@pytest.fixture(scope="module")
def pairs():
    found = {}
    for size, stem in ARMS.items():
        old = CONFIG_DIR / f"{stem}_ext30k.py"
        new = CONFIG_DIR / f"{stem}_ext45k.py"
        assert old.exists(), f"{old.name} is missing, so there is nothing to continue"
        assert new.exists(), f"{new.name} is missing"
        found[size] = (assignments(old), assignments(new))
    return found


def test_all_four_sizes_have_a_45k_config(pairs):
    """Plan ア is all four sizes. A missing one would quietly become plan イ."""
    assert set(pairs) == set(ARMS)


@pytest.mark.parametrize("size", sorted(ARMS))
def test_max_steps_is_45000(pairs, size):
    assert pairs[size][1]["max_steps"] == 45000


@pytest.mark.parametrize("size", sorted(ARMS))
def test_the_previous_config_stopped_at_30000(pairs, size):
    assert pairs[size][0]["max_steps"] == 30000


@pytest.mark.parametrize("size", sorted(ARMS))
def test_the_learning_rate_is_unchanged(pairs, size):
    """§3: the continuation runs at the same tenth, not a new rate."""
    old, new = pairs[size]
    assert new["learning_rate"] == old["learning_rate"]


@pytest.mark.parametrize("size", sorted(ARMS))
def test_the_schedule_is_constant_with_warmup(pairs, size):
    """The EXTEND gate refuses anything else, because HF decays over max_steps."""
    assert pairs[size][1]["lr_scheduler_type"] == "constant_with_warmup"


@pytest.mark.parametrize("size", sorted(ARMS))
def test_the_output_directory_is_a_new_one(pairs, size):
    old, new = pairs[size]
    assert "-ext45k" in new["model_path"]
    assert "-ext30k" not in new["model_path"]
    assert new["model_path"] != old["model_path"]


@pytest.mark.parametrize("size", sorted(ARMS))
def test_the_collapse_detector_is_armed(pairs, size):
    assert pairs[size][1]["degenerate_loss_threshold"] == 2.395


@pytest.mark.parametrize("size", sorted(ARMS))
def test_the_timeout_is_the_measured_one(pairs, size):
    """1,800 s. A wider window does not cure a rank divergence, it delays the report."""
    assert pairs[size][1]["ddp_timeout"] == 1800


@pytest.mark.parametrize("size", sorted(ARMS))
def test_the_seed_is_42_and_no_data_seed_is_declared(pairs, size):
    new = pairs[size][1]
    assert new["seed"] == 42
    assert "data_seed" not in new


@pytest.mark.parametrize("size", sorted(ARMS))
def test_nothing_else_drifted(pairs, size):
    """Only max_steps and model_path may differ from the config being continued."""
    old, new = pairs[size]
    assert set(old) == set(new), "a key was added or dropped, not just changed"
    differing = {k for k in old if old[k] != new[k]}
    assert differing == EXPECTED_DIFFERENCES
