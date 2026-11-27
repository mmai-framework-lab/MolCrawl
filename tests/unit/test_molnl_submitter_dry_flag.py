"""The molnl submitter refuses a near miss of its dry-run flag instead of ignoring it.

On 2026-11-27 ``DRY_RUN=1`` was typed in front of workflows/submit-molnl-batch-sweep.sh,
whose flag is ``DRY``. The shell set a variable nothing reads, the run was taken as a
real submission, and twelve arms of the grid started from a command meant only to print
them. Nothing in the output said so until the last line, which read "submitted" rather
than "would be submitted".

The repository carries both names -- seventeen submitters read ``DRY_RUN`` and six read
``DRY`` -- so the mistake is available to anyone who used another one last. These tests
hold the refusal in place, and hold it to firing only on that mistake.
"""

from __future__ import annotations

import os
import subprocess

import pytest

# The source is the tree this file sits in, not git's common directory: a worktree has
# its own copy, and reading the common directory's would test main instead of this.
REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUBMITTER = os.path.join(REPO, "workflows", "submit-molnl-batch-sweep.sh")
#: A deliberately bare environment: the guard must not depend on anything the caller
#: happens to export, and a DRY_RUN inherited from the test runner would hide a failure.
BARE = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp")}


def run(extra=None):
    env = dict(BARE)
    env.update(extra or {})
    return subprocess.run(["bash", SUBMITTER], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("name", ["DRY_RUN", "DRYRUN", "DRY_RUN_ONLY", "DRYRUNS"])
def test_a_near_miss_of_the_flag_stops_the_script(name):
    """Every spelling that starts with DRY and is not DRY refuses, naming the real flag."""
    done = run({name: "1"})
    assert done.returncode == 1, done.stdout + done.stderr
    assert name in done.stderr
    assert "DRY=1" in done.stderr
    # Nothing of the plan may have been printed: the refusal comes before any work.
    assert "would submit" not in done.stdout
    assert "submitted" not in done.stdout


def test_the_guard_does_not_fire_on_the_real_flag():
    """DRY itself is not a near miss, so the script goes on to its own checks."""
    done = run({"DRY": "1"})
    assert "reads only DRY" not in done.stderr
    # It stops at the first thing it genuinely needs, which is the input root.
    assert "LEARNING_SOURCE_DIR" in done.stderr


def test_the_guard_does_not_fire_when_no_such_variable_is_set():
    """With nothing DRY-like exported, the refusal is absent and the usual checks run."""
    done = run()
    assert "reads only DRY" not in done.stderr
    assert "LEARNING_SOURCE_DIR" in done.stderr


def test_the_refusal_is_the_first_thing_the_script_can_do():
    """The flag is checked before the required variables, so it cannot be masked.

    Were it placed after them, a missing root would answer first and the typed flag
    would go unmentioned -- the caller would fix the root, run again, and submit.
    """
    done = run({"DRY_RUN": "1"})
    assert "LEARNING_SOURCE_DIR" not in done.stderr
    assert "reads only DRY" in done.stderr
