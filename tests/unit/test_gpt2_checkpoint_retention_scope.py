"""Which checkpoints a nanoGPT run is entitled to delete.

cleanup_old_checkpoints ranks by step and keeps the newest few. An out_dir can
already hold a finished campaign -- runs of one subset at different window
lengths derive the same directory name -- and a foreign campaign carries the
higher step numbers, so it fills the keep window and the new run deletes its own
work instead. Once the new run passes those numbers the old campaign goes too.

Ownership is decided by lineage, not by step: a fresh run owns nothing that was
already on disk, a resumed run owns its own history from before the restart, and
both own whatever they write from then on.
"""
import importlib.util
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "molcrawl" / "models" / "gpt2" / "train.py"


def _load():
    """Import the two functions without running the training script.

    train.py executes at import; only the definitions are wanted here, so the
    source is trimmed to them.
    """
    text = _SRC.read_text()
    keep = []
    for name in ("def own_checkpoint_steps(", "def cleanup_old_checkpoints("):
        start = text.index(name)
        rest = text[start:]
        nxt = [rest.index(m) for m in ("\ndef ", "\nif __name__", "\n_WARNED") if m in rest]
        keep.append(rest[: min(nxt)] if nxt else rest)
    mod = types.ModuleType("_gpt2_retention")
    mod.__dict__.update({"glob": __import__("glob"), "os": __import__("os"),
                         "shutil": __import__("shutil"), "_WARNED_FOREIGN": set()})
    exec("\n\n".join(keep), mod.__dict__)
    return mod


g = _load()


def _make(tmp_path, steps):
    for s in steps:
        (tmp_path / f"checkpoint-{s}").mkdir()
    return str(tmp_path)


def _on_disk(tmp_path):
    return sorted(int(d.name.split("-")[1]) for d in tmp_path.iterdir()
                  if d.name.startswith("checkpoint-"))


def test_a_fresh_run_owns_nothing_that_was_already_there():
    assert g.own_checkpoint_steps("/nonexistent", None) == set()


def test_a_resume_owns_its_history_up_to_where_it_restarted(tmp_path):
    _make(tmp_path, [1000, 2000, 3000, 111230])

    own = g.own_checkpoint_steps(str(tmp_path), 3000)

    assert own == {1000, 2000, 3000}          # 111230 is ahead of us: not ours


def test_a_previous_campaign_survives_a_fresh_run(tmp_path):
    """The failure this exists to prevent: 3 old + 4 new, keep 3."""
    old, mine = [100000, 105000, 111230], [1000, 2000, 3000, 4000]
    d = _make(tmp_path, old + mine)

    g.cleanup_old_checkpoints(d, 3, own_steps=set(mine))

    survived = _on_disk(tmp_path)
    assert all(s in survived for s in old)
    assert [s for s in survived if s < 100000] == [2000, 3000, 4000]


def test_without_ownership_the_new_run_deletes_its_own_work(tmp_path):
    """Documents the behaviour being fixed; own_steps=None is unchanged."""
    old, mine = [100000, 105000, 111230], [1000, 2000, 3000, 4000]
    d = _make(tmp_path, old + mine)

    g.cleanup_old_checkpoints(d, 3)

    assert _on_disk(tmp_path) == sorted(old)   # every new checkpoint is gone


def test_a_resumed_run_still_prunes_what_it_wrote_before_the_restart(tmp_path):
    """Protecting too much would let checkpoints accumulate without bound."""
    d = _make(tmp_path, [1000, 2000, 3000, 4000, 5000])
    own = g.own_checkpoint_steps(d, 3000) | {4000, 5000}

    g.cleanup_old_checkpoints(d, 2, own_steps=own)

    assert _on_disk(tmp_path) == [4000, 5000]


def test_the_best_val_checkpoint_is_still_protected(tmp_path):
    d = _make(tmp_path, [1000, 2000, 3000, 4000])

    g.cleanup_old_checkpoints(d, 2, protect_step=1000,
                              own_steps={1000, 2000, 3000, 4000})

    assert _on_disk(tmp_path) == [1000, 3000, 4000]


def test_nothing_is_deleted_when_everything_is_foreign(tmp_path):
    old = [100000, 105000, 111230]
    d = _make(tmp_path, old)

    g.cleanup_old_checkpoints(d, 1, own_steps=set())

    assert _on_disk(tmp_path) == old


@pytest.mark.parametrize("limit", [None])
def test_no_limit_still_means_no_pruning(tmp_path, limit):
    d = _make(tmp_path, [1000, 2000, 3000])

    g.cleanup_old_checkpoints(d, limit, own_steps={1000, 2000, 3000})

    assert _on_disk(tmp_path) == [1000, 2000, 3000]
