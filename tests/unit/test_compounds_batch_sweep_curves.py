"""The grid's BERT reader: what it must not get wrong about an evaluation series.

The BERT arms' history lives in trainer_state.json, where the Trainer writes one entry
per evaluation and the breakdown callback adds a second at the same step. Every test
here is about a way that shape can be misread -- an entry counted as a point, a
resumed arm read from the wrong file, a running arm reported as finished, or a floor
measured for next-token prediction applied to a masked-token loss.

The states are written into a temporary directory, so the test says the same thing
after the 48 arms finish.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "compounds_batch_sweep_curves",
    os.path.join(os.path.abspath(SCRIPTS), "compounds_batch_sweep_curves.py"))
curves = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(curves)


ARM = {"arch": "bert", "size": "small", "batch": "256", "lr": "2.21e-04",
       "lr_slug": "2p21e4", "steps": "450000", "judge_metric": "eval_loss_mask",
       "out_dir": "compounds-small-b256-lr2p21e4"}


def write_state(runs, out_dir, step, history, global_step=None):
    """One checkpoint directory holding a trainer_state.json."""
    path = os.path.join(runs, out_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "trainer_state.json"), "w", encoding="utf-8") as fh:
        json.dump({"global_step": global_step if global_step is not None else step,
                   "max_steps": 450000, "log_history": history}, fh)
    return path


def pair(step, mask, loss, epoch=1.0):
    """What the Trainer and the breakdown callback write for one evaluation."""
    return [
        {"epoch": epoch, "eval_loss": loss, "eval_runtime": 5.0, "step": step},
        {"epoch": epoch, "eval_loss_mask": mask, "eval_loss_copy": 0.001,
         "eval_loss_random": mask + 0.006, "step": step},
    ]


def test_two_entries_at_one_step_are_one_evaluation_point(tmp_path):
    runs = str(tmp_path)
    write_state(runs, ARM["out_dir"], 4000,
                pair(2000, 0.30, 0.90) + pair(4000, 0.20, 0.60))
    _curve, summary, missing = curves.rows_for_bert([ARM], runs)
    assert missing == []
    assert summary[0]["eval_points"] == 2


def test_the_judged_metric_is_the_masked_one(tmp_path):
    runs = str(tmp_path)
    # eval_loss falls below eval_loss_mask at the later step. Judging on eval_loss
    # would pick step 4000; the judged metric picks step 2000.
    write_state(runs, ARM["out_dir"], 4000,
                pair(2000, 0.20, 0.90) + pair(4000, 0.30, 0.10))
    _curve, summary, _ = curves.rows_for_bert([ARM], runs)
    assert summary[0]["judged_metric"] == "eval_loss_mask"
    assert summary[0]["best_value"] == "0.200000"
    assert summary[0]["best_step"] == 2000


def test_the_second_series_is_carried_beside_it(tmp_path):
    runs = str(tmp_path)
    write_state(runs, ARM["out_dir"], 4000,
                pair(2000, 0.20, 0.90) + pair(4000, 0.30, 0.10))
    _curve, summary, _ = curves.rows_for_bert([ARM], runs)
    row = summary[0]
    assert row["beside_metric"] == "eval_loss"
    assert row["beside_at_best"] == "0.900000"   # at best_step, not the lowest
    assert row["beside_final"] == "0.100000"


def test_the_newest_checkpoint_is_the_one_read(tmp_path):
    runs = str(tmp_path)
    # A resumed arm keeps older checkpoints whose history stops early.
    write_state(runs, ARM["out_dir"], 2000, pair(2000, 0.30, 0.90))
    write_state(runs, ARM["out_dir"], 10000,
                pair(2000, 0.30, 0.90) + pair(10000, 0.10, 0.40))
    _curve, summary, _ = curves.rows_for_bert([ARM], runs)
    assert summary[0]["eval_points"] == 2
    assert summary[0]["final_step"] == 10000
    assert "checkpoint-10000" in summary[0]["source"]


def test_the_newest_is_by_step_not_by_name(tmp_path):
    runs = str(tmp_path)
    write_state(runs, ARM["out_dir"], 9000, pair(9000, 0.20, 0.50))
    write_state(runs, ARM["out_dir"], 10000, pair(10000, 0.10, 0.40))
    # Sorted as text, "checkpoint-9000" comes after "checkpoint-10000".
    assert "checkpoint-10000" in curves.newest_state(
        os.path.join(runs, ARM["out_dir"]))


def test_a_running_arm_reports_the_loop_ahead_of_its_last_evaluation(tmp_path):
    runs = str(tmp_path)
    write_state(runs, ARM["out_dir"], 10000,
                pair(10000, 0.10, 0.40), global_step=10750)
    _curve, summary, _ = curves.rows_for_bert([ARM], runs)
    assert summary[0]["steps_reached"] == 10750
    assert summary[0]["final_step"] == 10000
    assert summary[0]["steps_planned"] == 450000


def test_the_masked_token_floor_is_the_unigram_not_the_blended_figure(tmp_path):
    # eval_loss_mask is scored at the [MASK] positions alone, so its floor is the
    # unigram cross-entropy there (2.4733), not the blended figure (2.2242) that
    # belongs beside eval_loss, and not the arms' configured trip line (2.395).
    assert curves.BERT_FLOOR == 2.4733
    runs = str(tmp_path)
    write_state(runs, ARM["out_dir"], 2000, pair(2000, 2.50, 2.30))
    _curve, summary, _ = curves.rows_for_bert([ARM], runs)
    assert summary[0]["collapsed"] == "yes"


def test_an_arm_below_the_floor_is_not_called_collapsed(tmp_path):
    runs = str(tmp_path)
    # Where the xl arms sit: above the configured trip line's neighbourhood but
    # clearly below the unigram floor, which is learning, not collapse.
    write_state(runs, ARM["out_dir"], 2000, pair(2000, 1.95, 2.03))
    _curve, summary, _ = curves.rows_for_bert([ARM], runs)
    assert summary[0]["collapsed"] == "no"


def test_an_arm_with_no_checkpoint_is_named_not_skipped(tmp_path):
    _curve, summary, missing = curves.rows_for_bert([ARM], str(tmp_path))
    assert summary == []
    assert [m["out_dir"] for m in missing] == [ARM["out_dir"]]


def test_an_arm_whose_breakdown_never_landed_is_missing(tmp_path):
    runs = str(tmp_path)
    # eval_loss alone: the arm is judged on a metric the file does not carry, and
    # falling back to eval_loss would quietly report another quantity.
    write_state(runs, ARM["out_dir"], 2000,
                [{"epoch": 1.0, "eval_loss": 0.9, "step": 2000}])
    _curve, summary, missing = curves.rows_for_bert([ARM], runs)
    assert summary == []
    assert len(missing) == 1


def test_the_curve_rows_carry_both_series_and_the_token_count(tmp_path):
    runs = str(tmp_path)
    write_state(runs, ARM["out_dir"], 2000, pair(2000, 0.30, 0.90))
    curve, _summary, _ = curves.rows_for_bert([ARM], runs)
    metrics = sorted(r["metric"] for r in curve)
    assert metrics == ["eval_loss", "eval_loss_copy", "eval_loss_mask",
                       "eval_loss_random"]
    one = curve[0]
    assert one["rows_seen"] == 2000 * 256
    assert one["tokens_seen"] == 2000 * 256 * curves.SEQUENCE_LENGTH
    assert one["seed"] == curves.SEED


def test_gpt2_rows_leave_the_second_series_empty(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    arm = dict(ARM, arch="gpt2", steps="46800", judge_metric="val_loss")
    name = f"mc-cmp-gpt2-{arm['size']}-b{arm['batch']}-lr{arm['lr_slug']}-373000.out"
    (logs / name).write_text(
        "step 0: train loss 6.5190, val loss 6.5204\n"
        "step 160: train loss 0.5000, val loss 0.5500\n")
    _curve, summary, _ = curves.rows_for([arm], str(logs))
    row = summary[0]
    assert row["beside_metric"] == "" and row["beside_at_best"] == ""
    assert row["collapsed"] == "no"


def test_the_summary_tsv_holds_the_new_columns(tmp_path):
    runs = tmp_path / "runs"
    arms = tmp_path / "rt1-arms.tsv"
    with open(arms, "w", newline="", encoding="utf-8") as handle:
        handle.write("#! report-tsv 1\n")
        writer = csv.DictWriter(handle, fieldnames=list(ARM), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow(ARM)
    write_state(str(runs), ARM["out_dir"], 4000,
                pair(2000, 0.30, 0.90) + pair(4000, 0.20, 0.60))
    rc = curves.main(["--arms", str(arms), "--runs", str(runs), "--arch", "bert",
                      "--curves-out", str(tmp_path / "rt1-c.tsv"),
                      "--summary-out", str(tmp_path / "rt1-s.tsv")])
    assert rc == 0
    with open(tmp_path / "rt1-s.tsv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(
            [line for line in handle if not line.startswith("#")], delimiter="\t"))
    assert rows[0]["beside_metric"] == "eval_loss"
    assert rows[0]["judged_metric"] == "eval_loss_mask"
    # The arm ends at 0.20, far under the unigram floor at the [MASK] positions.
    assert rows[0]["collapsed"] == "no"


def test_bert_without_runs_is_refused(tmp_path):
    arms = tmp_path / "rt1-arms.tsv"
    arms.write_text("#! report-tsv 1\n")
    try:
        curves.main(["--arms", str(arms), "--arch", "bert", "--logs", str(tmp_path),
                     "--curves-out", str(tmp_path / "rt1-c.tsv"),
                     "--summary-out", str(tmp_path / "rt1-s.tsv")])
    except SystemExit as exit_:
        assert exit_.code != 0
    else:
        raise AssertionError("bert without --runs should be refused")
