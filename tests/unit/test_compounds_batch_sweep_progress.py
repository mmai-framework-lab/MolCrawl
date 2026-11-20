"""The grid's progress reader: it must not misattribute a log or a step.

A progress table is read by someone deciding whether to wait, so the two ways it can
mislead are an arm labelled with another arm's log and a step that is not the step the
loop is at. Both are pinned here, on logs written into a temporary directory rather
than on the live runs, so the test says the same thing after those runs finish.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "compounds_batch_sweep_progress",
    os.path.join(os.path.abspath(SCRIPTS), "compounds_batch_sweep_progress.py"))
progress = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(progress)


ARM = {"arch": "bert", "size": "small", "batch": "256", "lr": "2.21e-04",
       "lr_slug": "2p21e4", "steps": "450000", "judge_metric": "eval_loss_mask",
       "out_dir": "compounds-small-b256-lr2p21e4"}


def write_log(logs, stem, segment, body):
    path = os.path.join(logs, f"{stem}-s{segment}-{100000 + segment}.out")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    return path


def test_the_job_name_comes_from_the_four_columns_not_the_directory():
    assert progress.job_stem(ARM) == "mc-cmp-bert-small-b256-lr2p21e4"


def test_the_gpt2_xl_job_name_is_not_its_output_directory():
    arm = {"arch": "gpt2", "size": "xl", "batch": "256", "lr_slug": "5p06e4",
           "out_dir": "compounds-ex-large-b256-lr5p06e4"}
    assert progress.job_stem(arm) == "mc-cmp-gpt2-xl-b256-lr5p06e4"
    assert arm["out_dir"] not in progress.job_stem(arm)


def test_the_step_is_the_last_progress_line_not_the_largest(tmp_path):
    logs = str(tmp_path / "logs")
    runs = str(tmp_path / "runs")
    os.makedirs(logs)
    os.makedirs(runs)
    # A restart within a segment takes the bar back down. The arm is at 1,000.
    write_log(logs, progress.job_stem(ARM), 1,
              "9000/450000 [1:00:00<10:00:00,  1.00it/s]\n"
              "1000/450000 [0:10:00<20:00:00,  1.00it/s]\n")
    row, note = progress.row_for(ARM, logs, runs)
    assert note is None
    assert row["steps_reached"] == "1000"
    assert row["fraction_of_budget"] == f"{1000 / 450000:.4f}"


def test_an_evaluation_bar_is_not_read_as_the_arms_position(tmp_path):
    logs = str(tmp_path / "logs")
    os.makedirs(logs)
    # Each evaluation prints its own bar over the evaluation batches. It comes after
    # the training line it interrupts, so the newest line in the log is the wrong one.
    write_log(logs, progress.job_stem(ARM), 1,
              "3000/450000 [1:00:00<140:00:00,  1.00it/s]\n"
              "63/79 [0:00:10<0:00:03,  5.00it/s]\n")
    row, note = progress.row_for(ARM, logs, str(tmp_path))
    assert note is None
    assert row["steps_reached"] == "3000"
    assert row["segment_remaining_h"] == f"{140.0:.2f}"


def test_a_log_with_only_another_budget_is_reported_not_guessed(tmp_path):
    logs = str(tmp_path / "logs")
    os.makedirs(logs)
    write_log(logs, progress.job_stem(ARM), 1,
              "1000/225000 [0:10:00<20:00:00,  1.00it/s]\n")
    row, note = progress.row_for(ARM, logs, str(tmp_path))
    assert note is not None and "225000" in note
    assert row["steps_reached"] == ""


def test_the_best_value_is_the_last_improvement(tmp_path):
    logs = str(tmp_path / "logs")
    os.makedirs(logs)
    write_log(logs, progress.job_stem(ARM), 1,
              "eval_loss_mask improved to 0.2000 at step 2000; saving off the interval\n"
              "eval_loss_mask improved to 0.0856 at step 92000; saving off the interval\n"
              "{'eval_loss_mask': 0.0856, 'epoch': 59.03}\n"
              "92000/450000 [6:00:00<23:00:00,  4.00it/s]\n")
    row, _ = progress.row_for(ARM, logs, str(tmp_path))
    assert row["best_value"] == "0.0856"
    assert row["best_step"] == "92000"
    assert row["epoch"] == "59.03"


def test_the_newest_segment_log_is_the_one_read(tmp_path):
    logs = str(tmp_path / "logs")
    os.makedirs(logs)
    stem = progress.job_stem(ARM)
    first = write_log(logs, stem, 1, "1000/450000 [0:10:00<20:00:00,  1.00it/s]\n")
    second = write_log(logs, stem, 2, "5000/450000 [0:50:00<19:00:00,  1.00it/s]\n")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    row, _ = progress.row_for(ARM, logs, str(tmp_path))
    assert row["steps_reached"] == "5000"
    assert row["source"] == os.path.basename(second)


def test_another_arms_log_is_not_read(tmp_path):
    logs = str(tmp_path / "logs")
    os.makedirs(logs)
    write_log(logs, "mc-cmp-bert-small-b512-lr3p13e4", 1,
              "7000/225000 [1:00:00<20:00:00,  1.00it/s]\n")
    row, note = progress.row_for(ARM, logs, str(tmp_path))
    assert row["steps_reached"] == ""
    assert note is not None and "no log" in note


def test_a_missing_log_still_names_the_arm(tmp_path):
    row, note = progress.row_for(ARM, str(tmp_path), str(tmp_path))
    assert row["arm"] == "mc-cmp-bert-small-b256-lr2p21e4"
    assert row["steps_planned"] == "450000"
    assert row["judged_metric"] == "eval_loss_mask"
    assert note is not None


def test_checkpoints_are_counted_from_the_run_directory(tmp_path):
    runs = tmp_path / "runs" / ARM["out_dir"]
    for step in (2000, 4000, 92000):
        (runs / f"checkpoint-{step}").mkdir(parents=True)
    (runs / "run_manifest.json").write_text("{}")
    row, _ = progress.row_for(ARM, str(tmp_path), str(tmp_path / "runs"))
    assert row["checkpoints_held"] == "3"
    assert row["newest_checkpoint"] == "92000"


def test_an_unknown_remaining_time_is_an_empty_cell():
    assert progress.hours("?") == ""
    assert progress.hours("23:11:03") == f"{23 + 11 / 60 + 3 / 3600:.2f}"
    assert progress.hours("11:03") == f"{11 / 60 + 3 / 3600:.2f}"


def test_the_tsv_carries_one_row_per_arm(tmp_path):
    arms = tmp_path / "rt1-arms.tsv"
    columns = list(ARM.keys())
    with open(arms, "w", newline="", encoding="utf-8") as handle:
        handle.write("#! report-tsv 1\n")
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t",
                               lineterminator="\n")
        writer.writeheader()
        writer.writerow(ARM)
        other = dict(ARM, size="medium", lr_slug="6p2e5",
                     out_dir="compounds-medium-b256-lr6p2e5")
        writer.writerow(other)
        writer.writerow(dict(ARM, arch="gpt2"))
    out = tmp_path / "rt1-progress.tsv"
    logs = tmp_path / "logs"
    logs.mkdir()
    rc = progress.main(["--arms", str(arms), "--logs", str(logs),
                        "--runs", str(tmp_path), "--out", str(out), "--arch", "bert"])
    assert rc == 0
    with open(out, encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(
            [line for line in handle if not line.startswith("#")], delimiter="\t")]
    assert [r["size"] for r in rows] == ["medium", "small"]
