"""What the ladder cross-check must not get wrong.

1. The gap to the neighbouring size has to be taken within one modality and one
   architecture, over the sizes that are actually on the ladder. molecule_nat_lang BERT
   has no xl rung and rna BERT's xl is still running, so "the size below" is not simply
   the previous row of the file.
2. A GPT-2 run's adopted step comes from the checkpoint metadata, not from the logged
   curve: the logging CSV carries four decimals and ties several iterations at the same
   rounded value. Comparing by value would report a tie as agreement.
3. A rung is in the 42 only when the audit calls the run production and its
   best_model_checkpoint disagrees with the minimum. Either half alone is not enough.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from ladder_checkpoint_crosscheck import (  # noqa: E402
    DIRECTIVE_CLASSES,
    add_neighbours,
    gpt2_adopted,
    load_runs,
    off_grid_checkpoints,
    read_tsv,
    rung,
)


def bert_row(run, size, best_step, best_value, min_step, min_value,
             match="no", production=True, run_dir="/nowhere"):
    return {
        "modality": "rna", "arch": "bert", "run": run, "run_dir": run_dir,
        "size": size, "seed": 42, "steps_run": 120960, "judged_on": "eval_loss_mask",
        "best_model_checkpoint_step": best_step, "best_metric_recorded": best_value,
        "judged_min": {"step": min_step, "value": min_value},
        "match": match, "production": production,
    }


def spec(modality, arch, size, run, on_ladder="yes"):
    return {"modality": modality, "arch": arch, "size": size, "run": run,
            "on_ladder": on_ladder, "source": "test", "note": ""}


def test_bert_rung_reports_the_gap_between_adopted_and_minimum():
    row = rung(spec("rna", "bert", "medium", "rna-medium-lr3e4"),
               bert_row("rna-medium-lr3e4", "medium", 115300, 2.638232,
                        116100, 2.636773))
    assert row["adopted_step"] == 115300
    assert row["adopted_value"] == "2.638232"
    assert row["min_step"] == 116100
    assert row["diff"] == "0.001459"
    assert row["in_42"] == "yes"


def test_in_42_needs_both_production_and_a_disagreement():
    agreeing = rung(spec("rna", "bert", "small", "r"),
                    bert_row("r", "small", 500, 1.0, 500, 1.0, match="yes"))
    diagnostic = rung(spec("rna", "bert", "small", "r"),
                      bert_row("r", "small", 500, 1.1, 400, 1.0,
                               match="no", production=False))
    assert agreeing["in_42"] == "no"
    assert diagnostic["in_42"] == "no"


def test_neighbour_is_the_size_below_on_the_same_track():
    rows = [
        rung(spec("rna", "bert", "small", "a"),
             bert_row("a", "small", 1, 2.695726, 1, 2.692825)),
        rung(spec("rna", "bert", "medium", "b"),
             bert_row("b", "medium", 1, 2.638232, 1, 2.636773)),
        rung(spec("rna", "bert", "large", "c"),
             bert_row("c", "large", 1, 2.637298, 1, 2.634906)),
        # Not on the ladder: still running. It must not become large's neighbour.
        rung(spec("rna", "bert", "xl", "d", on_ladder="no"),
             bert_row("d", "xl", 1, 8.414764, 1, 8.414764, match="yes")),
    ]
    add_neighbours(rows)
    assert rows[0]["neighbor_size"] == ""
    assert rows[1]["neighbor_size"] == "small"
    assert rows[2]["neighbor_size"] == "medium"
    assert rows[2]["neighbor_diff"] == "-0.000934"
    assert rows[2]["neighbor_diff_error"] == "0.001459"
    assert rows[2]["own_diff_error"] == "0.002392"
    assert rows[3]["neighbor_size"] == ""


def test_neighbour_does_not_cross_architectures(tmp_path):
    rows = [
        rung(spec("rna", "gpt2", "small", "g"), None),
        rung(spec("rna", "bert", "medium", "b"),
             bert_row("b", "medium", 1, 2.638232, 1, 2.636773)),
    ]
    add_neighbours(rows)
    assert rows[1]["neighbor_size"] == ""


def test_gpt2_adopted_reads_the_checkpoint_metadata(tmp_path):
    for step, value in ((2400, 0.550681), (2500, 0.550657), (4674, 1.076)):
        directory = tmp_path / f"checkpoint-{step}"
        directory.mkdir()
        (directory / "training_args.json").write_text(
            json.dumps({"iteration": step, "val_loss": value,
                        "best_val_loss": value}), encoding="utf-8")
    step, value, seen = gpt2_adopted(str(tmp_path))
    assert (step, seen) == (2500, 3)
    assert abs(value - 0.550657) < 1e-12


def test_gpt2_rung_compares_steps_not_the_rounded_curve(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-2500").mkdir(parents=True)
    (run_dir / "checkpoint-2500" / "training_args.json").write_text(
        json.dumps({"iteration": 2500, "val_loss": 0.550657}), encoding="utf-8")
    run_row = {
        "modality": "compounds", "arch": "gpt2", "run": "r", "run_dir": str(run_dir),
        "size": "small", "seed": 42, "steps_run": 4674, "judged_on": "val_loss",
        # Four decimals in the CSV put 2400 first although 2500 is lower.
        "val_curve": {"min_value": 0.5507, "min_iter": 2400},
        "production": True, "match": "n/a (§2: not through this decision)",
    }
    row = rung(spec("compounds", "gpt2", "small", "r"), run_row)
    assert row["adopted_step"] == 2500
    assert row["min_step"] == 2400
    # The adopted checkpoint is at or below the rounded minimum, so the two disagree
    # only in the CSV's last place -- measured on compounds small, where step 2,400
    # reads 0.5507008 and step 2,500 reads 0.5506568.
    assert row["diff"].startswith("0.000000")
    assert "4 桁丸め" in row["diff"]
    assert row["in_42"].startswith("no")


def test_gpt2_rung_reports_a_real_disagreement_as_one(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-1000").mkdir(parents=True)
    (run_dir / "checkpoint-1000" / "training_args.json").write_text(
        json.dumps({"iteration": 1000, "val_loss": 0.60}), encoding="utf-8")
    run_row = {
        "modality": "rna", "arch": "gpt2", "run": "r", "run_dir": str(run_dir),
        "size": "small", "judged_on": "val_loss",
        "val_curve": {"min_value": 0.50, "min_iter": 900},
        "production": True, "match": "n/a (§2: not through this decision)",
    }
    assert rung(spec("rna", "gpt2", "small", "r"), run_row)["diff"] == "step が違う"


def test_off_grid_checkpoints_counts_what_save_on_improve_wrote():
    run_row = {"checkpoints_present": [1000, 2000, 2800, 3000, 3100]}
    assert off_grid_checkpoints(run_row, 1000) == 2
    assert off_grid_checkpoints(run_row, "改善時のみ（保持 10）") == ""
    assert off_grid_checkpoints({"checkpoints_present": []}, 1000) == ""


def test_missing_run_is_marked_not_dropped():
    row = rung(spec("rna", "bert", "xl", "nope"), None)
    assert row["run"] == "nope"
    assert "走査に無い run" in row["note"]


def test_load_runs_prefers_the_production_copy(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"runs": [
        {"modality": "rna", "run": "r", "arch": "bert", "run_dir": "/a",
         "production": False, "steps_run": 10},
        {"modality": "rna", "run": "r", "arch": "bert", "run_dir": "/b",
         "production": True, "steps_run": 20},
    ]}), encoding="utf-8")
    runs = load_runs(str(path))
    assert runs[("rna", "r")]["run_dir"] == "/b"


def test_ladder_spec_in_the_repo_parses_and_names_five_modalities():
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "scripts", "ladder_runs.tsv")
    rows = read_tsv(path)
    assert {row["modality"] for row in rows} == {
        "compounds", "molecule_nat_lang", "protein_sequence", "rna", "genome_sequence"}
    assert {row["on_ladder"] for row in rows} <= {"yes", "no"}
    for row in rows:
        assert row["arch"] in ("bert", "gpt2"), row
        assert row["source"], row
    named = {(m, r) for entries in DIRECTIVE_CLASSES.values() for m, r in entries}
    # A rung the directive classes as not learning must say so in its own row.
    for row in rows:
        if (row["modality"], row["run"]) in named:
            assert row["on_ladder"] == "no", row
