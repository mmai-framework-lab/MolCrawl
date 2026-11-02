"""What the zero-shot rescorer writes out, and which checkpoint it scores.

The runs moved between checkouts after they finished, so the absolute path in
``best_model_checkpoint`` often does not exist any more. The step in its
basename still does, and the directory is looked for under the run.

What this pins down is the other half: when the adopted checkpoint cannot be
resolved, the rescorer stops. It used to score the largest step instead, and
the output said ``checkpoint`` without saying that the checkpoint was the final
one rather than the adopted one.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "clinvar_zeroshot_bert_rescore.py"
_spec = importlib.util.spec_from_file_location("_clinvar_rescore", _SRC)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def _run(tmp_path, steps, best=None, state_extra=None):
    for step in steps:
        (tmp_path / f"checkpoint-{step}").mkdir()
    state = dict(state_extra or {})
    if best is not None:
        state["best_model_checkpoint"] = best
    (tmp_path / f"checkpoint-{max(steps)}" / "trainer_state.json").write_text(
        json.dumps(state))
    return str(tmp_path)


def test_the_adopted_step_comes_back_with_the_path(tmp_path):
    run = _run(tmp_path, (1000, 2000, 3000), best=str(tmp_path / "checkpoint-1000"))

    path, step = rs.adopted_checkpoint(run)

    assert path.endswith("checkpoint-1000")    # not the newest, which is 3000
    assert step == 1000


def test_a_path_from_another_checkout_resolves_by_its_step(tmp_path):
    """The field holds an absolute path into the worktree the run came from."""
    run = _run(tmp_path, (1000, 2000),
               best="/gone/MolCrawl-wt-genome/learning_source/run/checkpoint-1000")

    path, step = rs.adopted_checkpoint(run)

    assert path == str(tmp_path / "checkpoint-1000")
    assert step == 1000


def test_an_unnamed_adopted_checkpoint_stops_instead_of_taking_the_last(tmp_path):
    run = _run(tmp_path, (1000, 2000), state_extra={"log_history": []})

    with pytest.raises(SystemExit) as caught:
        rs.adopted_checkpoint(run)

    assert "best_model_checkpoint" in str(caught.value)
    assert "2000" in str(caught.value)


def test_a_named_checkpoint_that_is_not_on_disk_stops(tmp_path):
    run = _run(tmp_path, (2000,), best=str(tmp_path / "checkpoint-1000"))

    with pytest.raises(SystemExit) as caught:
        rs.adopted_checkpoint(run)

    assert "checkpoint-1000" in str(caught.value)


def test_a_run_with_no_checkpoint_stops(tmp_path):
    with pytest.raises(SystemExit):
        rs.adopted_checkpoint(str(tmp_path))


# ---------------------------------------------------------------------------
# What the per-variant file carries
#
# The first pass kept only the aggregate AUROCs, which cannot be resampled for
# an interval. The pass that fixed that kept a bare score, which cannot be
# joined to a variant and cannot be re-summed over a narrower range. Both cost
# 21 runs. These pin down the third shape so there is no fourth.
# ---------------------------------------------------------------------------

_CSV_HEADER = ("vcv_id,review_status,consequence,chrom,pos,ref,alt,"
               "reference_sequence,variant_sequence,ClinicalSignificance\n")


def _csv(tmp_path, *rows):
    path = tmp_path / "clinvar.csv"
    path.write_text(_CSV_HEADER + "".join(rows))
    return str(path)


def _row(vcv, chrom, pos, ref, alt, sig, length=8):
    seq = "A" * length
    var = seq[:length // 2] + alt + seq[length // 2 + 1:]
    return (f"{vcv},criteria_provided,missense_variant,{chrom},{pos},{ref},{alt},"
            f"{seq},{var},{sig}\n")


def test_rows_carry_what_identifies_the_variant(tmp_path):
    path = _csv(tmp_path, _row("VCV1", "21", 6116503, "C", "T", "Pathogenic"))

    got, = rs.read_rows(path, {"21"})

    assert got["vcv_id"] == "VCV1"
    assert got["chrom"] == "21"
    assert got["pos"] == 6116503          # an int, not the string from the CSV
    assert got["ref_allele"] == "C"
    assert got["alt_allele"] == "T"
    assert got["clinical_significance"] == "Pathogenic"
    assert got["label_pathogenic"] == 1


def test_a_conflicting_call_is_left_out_of_the_rows(tmp_path):
    path = _csv(tmp_path,
                _row("VCV1", "21", 1, "C", "T", "Conflicting classifications"),
                _row("VCV2", "21", 2, "C", "T", "Benign"))

    got = rs.read_rows(path, {"21"})

    assert [r["vcv_id"] for r in got] == ["VCV2"]
    assert got[0]["label_pathogenic"] == 0


def test_the_four_ranges_are_1024_257_65_and_one_position():
    assert rs.range_counts(1024, 512) == {"window": 1024, "flank128": 257,
                                          "flank32": 65, "point": 1}


def test_the_point_range_is_the_variant_position_alone():
    values = [float(i) for i in range(1024)]

    got = rs.range_means(values, at=512)

    assert got["point"] == 512.0
    assert got["flank32"] == sum(values[480:545]) / 65
    assert got["flank128"] == sum(values[384:641]) / 257
    assert got["window"] == sum(values) / 1024


def test_a_range_that_runs_off_the_window_is_clipped():
    values = [1.0] * 10

    assert rs.range_slice(10, 1, 128) == (0, 10)
    assert rs.range_means(values, at=1)["flank128"] == 1.0


def test_the_window_range_uses_the_total_the_device_reduced():
    """The whole-window value has to be the number the earlier runs reported.

    Summing the per-position floats again in Python would add them in another
    order, so the window mean is the device's reduction divided by the count.
    """
    values = [1.0] * 4

    got = rs.range_means(values, at=2, total=8.0)

    assert got["window"] == 2.0           # 8.0 / 4, not sum(values) / 4
    assert got["point"] == 1.0


def test_leaving_the_per_variant_scores_out_is_not_an_option():
    """The flag that skipped them is gone; two passes were paid for want of rows."""
    source = _SRC.read_text()

    assert "no-predictions" not in source
    assert "no_predictions" not in source
