"""Which checkpoints a nanoGPT run keeps, once it can tell them apart.

Newest-first is the wrong rule for a ladder. The checkpoint a size is scored at
is its best one, and on a curve that turns -- the protein GPT-2 large arm
bottomed at step 1,850 of 4,674 -- that is not among the last few. BERT has
ranked on the judged metric since _checkpoint_retention; this is the same rule on
the GPT-2 side.

The score has to come from the run's own evaluations. A checkpoint directory
records ``best_val_loss``, which is the running best and therefore the same
number in every directory written after the minimum: it cannot order them.
"""
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "molcrawl" / "models" / "gpt2" / "train.py"


def _load():
    """Import the definitions without running the training script."""
    text = _SRC.read_text()
    keep = []
    for name in ("def own_checkpoint_steps(", "def eval_history(",
                 "def cleanup_old_checkpoints("):
        start = text.index(name)
        rest = text[start:]
        nxt = [rest.index(m) for m in ("\ndef ", "\nif __name__", "\n_WARNED") if m in rest]
        keep.append(rest[: min(nxt)] if nxt else rest)
    mod = types.ModuleType("_gpt2_ranking")
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


def _log(tmp_path, name, rows):
    lines = ["iter, train_loss, val_loss"]
    lines += [f"{i}, {t:.4f}, {v:.4f}" for i, t, v in rows]
    (tmp_path / name).write_text("\n".join(lines) + "\n")


# --- the policy without scores is the one that was there before ------------- #

def test_without_scores_it_still_keeps_the_newest(tmp_path):
    _make(tmp_path, [1000, 2000, 3000, 4000, 5000])

    g.cleanup_old_checkpoints(str(tmp_path), 2)

    assert _on_disk(tmp_path) == [4000, 5000]


# --- with scores, the best survive ------------------------------------------ #

def test_the_best_checkpoints_survive_a_curve_that_turns(tmp_path):
    """The minimum is at 2000, five saves back. Newest-first would delete it."""
    _make(tmp_path, [1000, 2000, 3000, 4000, 5000, 6000, 7000])
    scores = {1000: 3.0, 2000: 2.1, 3000: 2.4, 4000: 2.6, 5000: 2.8, 6000: 3.1, 7000: 3.4}

    g.cleanup_old_checkpoints(str(tmp_path), 2, scores=scores, keep_latest=1)

    assert _on_disk(tmp_path) == [2000, 3000, 7000]   # best two, plus newest


def test_keep_latest_is_for_resume_and_is_kept_even_when_it_is_the_worst(tmp_path):
    _make(tmp_path, [1000, 2000, 3000])
    scores = {1000: 1.0, 2000: 1.1, 3000: 9.9}

    g.cleanup_old_checkpoints(str(tmp_path), 1, scores=scores, keep_latest=1)

    assert _on_disk(tmp_path) == [1000, 3000]


def test_a_checkpoint_with_no_score_cannot_be_ranked_and_goes(tmp_path):
    """Saved off the eval grid, so nothing places it against the others."""
    _make(tmp_path, [1000, 1500, 2000, 3000])
    scores = {1000: 1.0, 2000: 1.1, 3000: 1.2}      # 1500 was never evaluated

    g.cleanup_old_checkpoints(str(tmp_path), 2, scores=scores, keep_latest=1)

    assert _on_disk(tmp_path) == [1000, 2000, 3000]


def test_the_protected_step_survives_either_policy(tmp_path):
    _make(tmp_path, [1000, 2000, 3000])
    scores = {1000: 9.9, 2000: 1.1, 3000: 1.2}

    g.cleanup_old_checkpoints(str(tmp_path), 1, protect_step=1000,
                              scores=scores, keep_latest=1)

    assert _on_disk(tmp_path) == [1000, 2000, 3000]


def test_ownership_still_bounds_what_may_be_deleted(tmp_path):
    """A finished campaign in the same out_dir is not this run's to prune."""
    _make(tmp_path, [1000, 2000, 100000, 105000])
    scores = {1000: 1.0, 2000: 1.1}

    g.cleanup_old_checkpoints(str(tmp_path), 1, own_steps={1000, 2000},
                              scores=scores, keep_latest=1)

    assert _on_disk(tmp_path) == [1000, 2000, 100000, 105000]


# --- the history a resume ranks on ------------------------------------------ #

def test_eval_history_reads_back_what_the_run_measured(tmp_path):
    _log(tmp_path, "logging_20260101_000000.csv", [(100, 4.0, 3.9), (200, 3.5, 3.4)])

    assert g.eval_history(str(tmp_path)) == {100: 3.9, 200: 3.4}


def test_a_resume_inherits_the_earlier_segment(tmp_path):
    """Two segments, two CSVs. Without this the first segment has no score and
    cleanup deletes it for being unrankable."""
    _log(tmp_path, "logging_20260101_000000.csv", [(100, 4.0, 3.9), (200, 3.5, 3.4)])
    _log(tmp_path, "logging_20260102_000000.csv", [(300, 3.2, 3.1)])

    assert g.eval_history(str(tmp_path)) == {100: 3.9, 200: 3.4, 300: 3.1}


def test_a_line_cut_off_by_the_time_limit_does_not_stop_the_run(tmp_path):
    (tmp_path / "logging_20260101_000000.csv").write_text(
        "iter, train_loss, val_loss\n100, 4.0000, 3.9000\n200, 3.50"
    )

    assert g.eval_history(str(tmp_path)) == {100: 3.9}


def test_no_logs_is_an_empty_history_not_an_error(tmp_path):
    assert g.eval_history(str(tmp_path)) == {}


# --- the audit that reads those logs back ----------------------------------- #

def _audit():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts" / "audit_legacy_ckpt.py"
    spec = importlib.util.spec_from_file_location("_audit_legacy_ckpt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_minimum_tied_at_log_precision_reports_both_steps(tmp_path):
    """compounds gpt2-small printed 0.5944 at 1500 and at 1550. The run compared
    in full precision and took 1550; calling 1500 "the" minimum would mark a
    correct ckpt.pt stale."""
    _log(tmp_path, "logging_20260821_112610.csv",
         [(1450, 0.5888, 0.5963), (1500, 0.5885, 0.5944), (1550, 0.5847, 0.5944)])

    steps, val = _audit().min_eval(str(tmp_path))

    assert steps == [1500, 1550]
    assert val == 0.5944


def test_a_single_minimum_is_a_single_step(tmp_path):
    _log(tmp_path, "logging_20260101_000000.csv",
         [(100, 4.0, 3.9), (200, 3.5, 3.4), (300, 3.3, 3.6)])

    assert _audit().min_eval(str(tmp_path)) == ([200], 3.4)
