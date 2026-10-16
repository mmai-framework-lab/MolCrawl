"""What a failed resume does, for each kind of failure.

The branch this covers used to restart pretraining from scratch when a checkpoint
would not load. A run cut by the time limit and continued is the ordinary case for
the larger sizes, so such a restart would train randomly initialised weights under
the same name and the run would read as complete.

The cases below are the actual messages: two were produced by corrupting a real
checkpoint (jobs 178180 and 178184), and the weights-only pair is what torch reports
when its restricted unpickler refuses a payload.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from molcrawl.models.bert.main import resume_failure_action  # noqa: E402

CKPT = "/somewhere/compounds-small-lr1e3/checkpoint-3200"


def test_nothing_was_being_resumed():
    """Without a checkpoint the failure is about something else."""
    assert resume_failure_action(RuntimeError("boom"), None, "") == "reraise"
    assert resume_failure_action(RuntimeError("boom"), "", "") == "reraise"


def test_corrupt_safetensors_reraises():
    """Measured in job 178180: the header error names none of the keywords."""
    exc = Exception("Error while deserializing header: HeaderTooLarge")
    assert resume_failure_action(exc, CKPT, "") == "reraise"


def test_corrupt_optimizer_pickle_reraises():
    """Measured in job 178184."""
    exc = Exception("invalid load key, 'n'.")
    assert resume_failure_action(exc, CKPT, "") == "reraise"


def test_weights_only_failure_without_pretrain_weights_stops():
    """The case that used to restart pretraining from random weights."""
    exc = Exception(
        "Weights only load failed. Re-running `torch.load` with `weights_only` set to "
        "`False` will likely succeed. WeightsUnpickler error: Unsupported global"
    )
    assert resume_failure_action(exc, CKPT, "") == "stop"


def test_trainer_checkpoint_message_without_pretrain_weights_stops():
    exc = Exception("Can't find a valid checkpoint at /somewhere/checkpoint-3200")
    assert resume_failure_action(exc, CKPT, "") == "stop"


def test_same_failure_restarts_a_finetune(tmp_path):
    """A fine-tune has weights to fall back to, so it may start over from them."""
    pre = tmp_path / "pretrained"
    (pre / "checkpoint-15000").mkdir(parents=True)
    exc = Exception("Weights only load failed. WeightsUnpickler error")
    assert resume_failure_action(exc, CKPT, str(pre)) == "restart_finetune"


def test_a_pretrain_path_that_does_not_exist_is_not_a_fallback():
    """A configured but missing path must not be read as weights to return to."""
    exc = Exception("Weights only load failed. WeightsUnpickler error")
    assert resume_failure_action(exc, CKPT, "/no/such/directory") == "stop"


def test_keyword_match_is_case_insensitive():
    exc = Exception("WEIGHTS ONLY LOAD FAILED")
    assert resume_failure_action(exc, CKPT, "") == "stop"
