"""The eight rna extension configs carry what the 2026-11-18 judgment asked for.

rna-order-2026-11-18b runs eight points to 241,920 steps; rna-judgment-2026-11-18b
fixes the schedule at each one. The values that matter are easy to get wrong by one
character and expensive to get wrong -- the xl arm alone is about 3,000 GPU hours --
and two of them (BERT's rate and warmup) are discarded at runtime unless the resume
copy is prepared as well, so a config that reads correctly is necessary but not
sufficient. What is checkable here is checked here.

The eighteen-epoch target is the nine-epoch schedule doubled: 120,960 steps is
13,440 per epoch, so 241,920 is exactly eighteen.
"""

import importlib
import sys

import pytest

PKG = "molcrawl.tasks.pretrain.configs.rna"

NINE_EPOCH_STEPS = 120960
EIGHTEEN_EPOCH_STEPS = 241920
STEPS_PER_EPOCH = 13440

# (extension config, the arm it continues, size, that arm's peak rate)
BERT_ARMS = [
    ("bert_small_lr3e4_ext242k", "bert_small_lr3e4", "small", 3e-4),
    ("bert_medium_lr3e4_ext242k", "bert_medium_lr3e4", "medium", 3e-4),
    ("bert_large_lr1p7e4_ext242k", "bert_large_lr1p7e4", "large", 1.7e-4),
    ("bert_xl_lr2p5e5_ext242k", "bert_xl_lr2p5e5", "xl", 2.5e-5),
]

GPT2_ARMS = [
    ("gpt2_small_lr1p2e3_9ep_ext242k", "gpt2_small_lr1p2e3_9ep", 1.2e-3),
    ("gpt2_medium_lr1p2e3_9ep_ext242k", "gpt2_medium_lr1p2e3_9ep", 1.2e-3),
    ("gpt2_large_lr6e4_9ep_ext242k", "gpt2_large_lr6e4_9ep", 6e-4),
    ("gpt2_xl_lr6e4_9ep_ext242k", "gpt2_xl_lr6e4_9ep", 6e-4),
]


@pytest.fixture(scope="module", autouse=True)
def runs_root(tmp_path_factory):
    """The GPT-2 configs refuse to import without a runs root, by design."""
    root = tmp_path_factory.mktemp("runs")
    import os

    before = os.environ.get("MODEL_OUTPUT_ROOT")
    os.environ["MODEL_OUTPUT_ROOT"] = str(root)
    yield str(root)
    if before is None:
        os.environ.pop("MODEL_OUTPUT_ROOT", None)
    else:
        os.environ["MODEL_OUTPUT_ROOT"] = before


def load(name):
    """Import fresh: these configs read the environment at import time."""
    sys.modules.pop(f"{PKG}.{name}", None)
    return importlib.import_module(f"{PKG}.{name}")


def test_eighteen_epochs_is_nine_doubled():
    assert EIGHTEEN_EPOCH_STEPS == 2 * NINE_EPOCH_STEPS
    assert NINE_EPOCH_STEPS == 9 * STEPS_PER_EPOCH
    assert EIGHTEEN_EPOCH_STEPS == 18 * STEPS_PER_EPOCH


@pytest.mark.parametrize("ext,arm,size,peak", BERT_ARMS)
def test_bert_extension_schedule(ext, arm, size, peak):
    a, e = load(arm), load(ext)
    assert a.learning_rate == pytest.approx(peak, rel=1e-12), "the arm is not the rate named here"
    assert a.max_steps == NINE_EPOCH_STEPS
    assert e.max_steps == EIGHTEEN_EPOCH_STEPS
    # No decay length separate from max_steps, so the extension has to drop the decay.
    assert e.lr_scheduler_type == "constant_with_warmup"
    assert e.learning_rate == pytest.approx(peak / 10, rel=1e-12)
    # The ramp is at the resume point, not a fraction of the run.
    assert e.warmup_steps == 500
    assert e.early_stopping is False
    assert e.seed == 42


@pytest.mark.parametrize("ext,arm,size,peak", BERT_ARMS)
def test_bert_extension_writes_somewhere_else(ext, arm, size, peak):
    a, e = load(arm), load(ext)
    assert e.model_path != a.model_path, "running in place lets retention prune the 9-epoch record"
    assert e.model_path.endswith("-ext242k")
    assert a.model_path in e.model_path or size in e.model_path


@pytest.mark.parametrize("ext,arm,size,peak", BERT_ARMS)
def test_bert_extension_declares_early_stopping_off(ext, arm, size, peak):
    """Same reason as the GPT-2 arms: the default is on, so it has to be stated."""
    e = load(ext)
    assert "early_stopping" in vars(e)
    assert e.early_stopping is False


@pytest.mark.parametrize("ext,arm,size,peak", BERT_ARMS)
def test_bert_extension_keeps_the_selection_rule(ext, arm, size, peak):
    """§4 wants the minimum kept, which means saving on improvement of the judged metric."""
    e = load(ext)
    assert e.save_on_improve is True
    for key in ("judge_on", "metric_for_best_model"):
        value = getattr(e, key, None)
        assert value in (None, "eval_loss_mask"), f"{key}={value!r} is not what §1 ranks on"


@pytest.mark.parametrize("ext,arm,peak", GPT2_ARMS)
def test_gpt2_extension_holds_the_decay_length(ext, arm, peak):
    a, e = load(arm), load(ext)
    assert a.max_iters == NINE_EPOCH_STEPS
    assert e.max_iters == EIGHTEEN_EPOCH_STEPS
    # The whole point of case A: get_lr reads the iter number, so holding
    # lr_decay_iters keeps every iter below 120,960 at the rate it saw before.
    assert e.lr_decay_iters == NINE_EPOCH_STEPS
    assert e.warmup_iters == a.warmup_iters == 2419
    assert e.learning_rate == pytest.approx(peak, rel=1e-12)
    assert e.min_lr == pytest.approx(peak / 10, rel=1e-12)
    assert e.seed == 42


@pytest.mark.parametrize("ext,arm,peak", GPT2_ARMS)
def test_gpt2_extension_resumes_into_its_own_tree(ext, arm, peak):
    a, e = load(arm), load(ext)
    # iter_num is only carried over on the resume path; the pretrain_dir path resets
    # it to 0 and drops the optimizer, which would restart the schedule.
    assert e.init_from == "resume"
    assert e.out_dir != a.out_dir, "resume reads out_dir/ckpt.pt and overwrites it"
    assert e.out_dir.endswith("-ext242k")
    assert e.tensorboard_dir == e.out_dir


@pytest.mark.parametrize("ext,arm,peak", GPT2_ARMS)
def test_gpt2_extension_does_not_stop_early(ext, arm, peak):
    """Stated in the config, not left to a default or to a launcher flag.

    train.py defaults early_stopping to True and the medium/large/xl bases set it
    True with patience 5, so a config that simply says nothing is a config that
    stops early. The first version of this test read the value with a False
    default and so passed the small arm, which declares nothing -- the absence of
    the name is exactly the failure being tested for. Require the declaration.
    """
    e = load(ext)
    assert "early_stopping" in vars(e), "the framework default is True; say it here"
    assert e.early_stopping is False


def test_rewrite_scheduler_writes_what_the_launcher_checks(tmp_path):
    """The staging rewrite and the bert-grid guard have to agree on the contract.

    workflows/bert-grid.sbatch accepts a scheduler.pt whose base_lrs equal the
    config's learning_rate and whose last_epoch is 0, because that file is this
    config's own schedule read from its first step. The file is kept rather than
    deleted so that optimizer.pt loads at all: transformers 4.45.1 guards both
    loads with one condition on scheduler.pt's existence (trainer.py:3204), so
    removing it leaves the Adam moments unloaded -- measured 2026-11-18, job
    363888, 42 optimizer state entries with the file present against 0 without.
    """
    torch = pytest.importorskip("torch")
    from scripts.prepare_rna_extension import rewrite_scheduler

    path = tmp_path / "scheduler.pt"
    torch.save(
        {
            "base_lrs": [2.5e-05, 2.5e-05],
            "last_epoch": 120960,
            "_step_count": 120961,
            "_is_initial": False,
            "_get_lr_called_within_step": False,
            "_last_lr": [0.0, 0.0],
            "lr_lambdas": [{}, {}],
        },
        path,
    )

    before = rewrite_scheduler(str(path), 2.5e-06)
    assert before["base_lrs"] == [2.5e-05, 2.5e-05]
    assert before["last_epoch"] == 120960

    after = torch.load(path, map_location="cpu", weights_only=False)
    assert after["base_lrs"] == [2.5e-06, 2.5e-06]
    assert after["last_epoch"] == 0
    assert after["_step_count"] == 1
    assert after["_last_lr"] == [0.0, 0.0]
    assert after["_is_initial"] is True
    # The lambdas are rebuilt by the new config, so the stored placeholders stay.
    assert after["lr_lambdas"] == [{}, {}]
