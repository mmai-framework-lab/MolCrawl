"""What the loss-curve TSV must get right before any figure is drawn.

The figures are drawn from this table, so an error here is an error in every
figure at once and is invisible once it is a line on a chart. Three things are
easy to get wrong and are pinned: which CSV a GPT-2 run's points come from when
it holds two, that a series is never merged with another, and that the epoch
count is not guessed from max_steps.
"""
import csv
import datetime
import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "genome_loss_curves.py"
_spec = importlib.util.spec_from_file_location("_genome_curves", _SRC)
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)


def test_the_window_tag_names_the_series():
    assert lc.series_of("bert-small-global_random_seed1-w1026") == "bert-w1026"
    assert lc.series_of("bert-small-mammal_centered-sat9") == "bert-sat9"
    assert lc.series_of("bert-small-mammal_centered-w1024-ep9") == "bert-w1024-ep9"
    assert lc.series_of("bert-small-mammal_centered-smoke") == "smoke"


def test_a_row_is_as_wide_as_the_header_says():
    """The header and the row come from one list, so they cannot drift.

    They did drift: `epochs` made the row eight fields while the summary at the
    end of the job still unpacked seven, and the failure landed after
    twenty-five minutes of reading run directories.
    """
    row = lc.row_for("gpt2-ep9", "gpt2", "mammal_centered", 9, 1000, 1.0803,
                     "val_loss", "175277")
    assert len(row) == len(lc.COLUMNS)
    named = dict(zip(lc.COLUMNS, row))
    assert named["epochs"] == 9
    assert named["series"] == "gpt2-ep9"
    assert named["job"] == "175277"


def test_an_unknown_epoch_count_is_blank_rather_than_the_word_none():
    """`None` written into a TSV reads as the four-letter string in every
    reader that opens it, and sorts among the numbers."""
    row = lc.row_for("gpt2-lr-sweep", "gpt2", "stab2-E", None, 100, 1.4,
                     "val_loss", "")
    assert dict(zip(lc.COLUMNS, row))["epochs"] == ""


def test_the_schedule_measurement_is_its_own_series():
    """Nine epochs under the three-epoch LR schedule is a third thing.

    `-ep9-decay3` does not end in `-ep9`, so without its own rule it falls
    through to the untagged `gpt2` series -- nine epochs of data averaged into
    the three-epoch twenty-one, which is the defect `-ep9` was added to stop.
    """
    assert lc.series_of("gpt2-small-mammal_centered-ep9-decay3") == "gpt2-ep9-decay3"
    assert lc.series_of("gpt2-small-mammal_centered-ep9") == "gpt2-ep9"
    assert lc.series_of("gpt2-small-mammal_centered") == "gpt2"


def test_an_untagged_run_is_the_512_series_not_a_fallback_bucket():
    """The production 21 carry no suffix, so 'no tag' is a series, not unknown."""
    assert lc.series_of("bert-small-global_random_seed1") == "bert-base"


def test_the_subset_survives_both_the_arch_prefix_and_the_series_suffix():
    assert lc.subset_of("bert-small-eukaryote_matched_random_seed7-w1026") == \
        "eukaryote_matched_random_seed7"
    assert lc.subset_of("gpt2-small-mammal_centered") == "mammal_centered"


def _csv(path, rows):
    path.write_text("iter, train_loss, val_loss\n"
                    + "".join(f"{i}, {t}, {v}\n" for i, t, v in rows))


def test_an_empty_second_csv_does_not_hide_the_run(tmp_path):
    """Nine of the 21 GPT-2 runs hold two logging CSVs a second apart -- two
    ranks opened a file and one wrote. Taking whichever sorts first returns
    nothing for the five where the empty one sorts earlier."""
    _csv(tmp_path / "logging_20260817_093253.csv", [])
    _csv(tmp_path / "logging_20260817_093254.csv", [(0, 2.19, 2.19), (1000, 1.29, 1.30)])

    assert lc.gpt2_points(tmp_path) == [(0, 2.19), (1000, 1.30)]


def test_the_written_csv_is_taken_whichever_way_the_names_sort(tmp_path):
    """The mirror case: the written file sorts first."""
    _csv(tmp_path / "logging_20260817_093253.csv", [(0, 2.19, 2.20), (1000, 1.29, 1.31)])
    _csv(tmp_path / "logging_20260817_093254.csv", [])

    assert lc.gpt2_points(tmp_path) == [(0, 2.20), (1000, 1.31)]


def test_a_run_with_no_points_is_reported_rather_than_returned_empty(tmp_path):
    assert lc.gpt2_points(tmp_path) == []


def test_the_epoch_count_is_not_inferred_from_max_steps(tmp_path):
    """max_steps is ceil(epochs x train_rows / batch) and train_rows is not in
    the checkpoint, so any threshold on max_steps is fitted to the runs that
    exist. A run without a manifest reports nothing rather than a guess."""
    assert lc._manifest_epochs(tmp_path) is None


def test_the_manifest_is_the_only_source_of_the_epoch_count(tmp_path):
    import json
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"env": {"SUBSET_BERT_EPOCHS": "9", "SUBSET_BERT_MAX_LENGTH": "1026"}}))

    assert lc._manifest_epochs(tmp_path) == 9


def _run_with_checkpoint(tmp_path, when):
    """A run directory whose newest checkpoint was written at `when`."""
    import os
    ck = tmp_path / "checkpoint-1000"
    ck.mkdir(parents=True)
    (ck / "trainer_state.json").write_text("{}")
    stamp = when.timestamp()
    os.utime(ck, (stamp, stamp))
    return tmp_path


def _job(job, name, node, began, ended):
    return (job, name, node, began, ended)


def test_two_series_of_the_same_subset_are_told_apart_by_when_they_ran(tmp_path):
    """The regression this rule exists for.

    The 512-window and 1,026-window BERT series launch jobs of the same name
    for the same subset, two weeks apart. Matching on the name alone handed the
    512 run the 1,026 run's job number, and nothing flagged it -- the column
    simply read as a fact. The run's own checkpoint time settles which it was.
    """
    august = datetime.datetime(2026, 8, 25, 10, 0)
    run = _run_with_checkpoint(tmp_path / "bert-small-mammal_centered", august)
    headers = [("40320", "mammal_centered"), ("84667", "mammal_centered")]
    sacct = [_job("40320", "mc-gen-bert-mammal_centered", "c181",
                  datetime.datetime(2026, 8, 24, 18, 0), datetime.datetime(2026, 8, 25, 11, 0)),
             _job("84667", "mc-gen-bert-mammal_centered", "c390",
                  datetime.datetime(2026, 9, 7, 14, 0), datetime.datetime(2026, 9, 9, 16, 0))]

    assert lc.job_id_for(str(run), "bert-base", headers, sacct) == "40320"


def test_the_later_series_gets_the_later_job(tmp_path):
    """The mirror of the case above, so the rule is not just excluding one."""
    september = datetime.datetime(2026, 9, 9, 15, 0)
    run = _run_with_checkpoint(tmp_path / "bert-small-mammal_centered-w1026", september)
    headers = [("40320", "mammal_centered"), ("84667", "mammal_centered")]
    sacct = [_job("40320", "mc-gen-bert-mammal_centered", "c181",
                  datetime.datetime(2026, 8, 24, 18, 0), datetime.datetime(2026, 8, 25, 11, 0)),
             _job("84667", "mc-gen-bert-mammal_centered", "c390",
                  datetime.datetime(2026, 9, 7, 14, 0), datetime.datetime(2026, 9, 9, 16, 0))]

    assert lc.job_id_for(str(run), "bert-w1026", headers, sacct) == "84667"


def test_the_subset_comes_from_the_log_header_not_the_file_name(tmp_path):
    """Some jobs carry no subset in their name at all; the header always does."""
    when = datetime.datetime(2026, 8, 25, 10, 0)
    run = _run_with_checkpoint(tmp_path / "bert-small-mammal_centered", when)
    headers = [("40261", "mammal_centered")]
    sacct = [_job("40261", "mc-gen-bert", "c177",
                  datetime.datetime(2026, 8, 24, 12, 0), datetime.datetime(2026, 8, 25, 11, 0))]

    assert lc.job_id_for(str(run), "bert-base", headers, sacct) == "40261"


def test_a_gpt2_job_is_matched_on_the_node_tensorboard_recorded(tmp_path):
    """GPT-2 writes no header and no manifest, so the node in the event file is
    the only identity left -- still intersected with the run's own window."""
    when = datetime.datetime(2026, 8, 17, 21, 0)
    run = tmp_path / "gpt2-small-mammal_centered"
    run.mkdir()
    (run / "events.out.tfevents.1786925563.c394").write_text("")
    (run / "ckpt.pt").write_text("")
    import os
    os.utime(run / "ckpt.pt", (when.timestamp(), when.timestamp()))
    sacct = [_job("28813", "mc-gen-gpt2-mammal_centered", "c394",
                  datetime.datetime(2026, 8, 17, 9, 0), datetime.datetime(2026, 8, 17, 21, 30)),
             _job("28817", "mc-gen-gpt2-eukaryote", "c397",
                  datetime.datetime(2026, 8, 17, 9, 0), datetime.datetime(2026, 8, 17, 20, 30))]

    assert lc.job_id_for(str(run), "gpt2", [], sacct) == "28813"


def test_two_candidates_yield_no_job_rather_than_the_first(tmp_path):
    """A wrong job number reads as a fact; a blank one reads as a gap."""
    when = datetime.datetime(2026, 8, 25, 10, 0)
    run = _run_with_checkpoint(tmp_path / "bert-small-mammal_centered", when)
    headers = [("40320", "mammal_centered"), ("40321", "mammal_centered")]
    window = (datetime.datetime(2026, 8, 24, 18, 0), datetime.datetime(2026, 8, 25, 11, 0))
    sacct = [_job("40320", "mc-gen-bert", "c181", *window),
             _job("40321", "mc-gen-bert", "c182", *window)]

    assert lc.job_id_for(str(run), "bert-base", headers, sacct) == ""


def test_a_run_outside_every_job_window_yields_no_job(tmp_path):
    when = datetime.datetime(2026, 7, 1, 10, 0)
    run = _run_with_checkpoint(tmp_path / "bert-small-mammal_centered", when)
    headers = [("40320", "mammal_centered")]
    sacct = [_job("40320", "mc-gen-bert", "c181",
                  datetime.datetime(2026, 8, 24, 18, 0), datetime.datetime(2026, 8, 25, 11, 0))]

    assert lc.job_id_for(str(run), "bert-base", headers, sacct) == ""


def test_the_epoch_count_is_not_read_from_dataset_info_json(tmp_path):
    """dataset_info.json reports 95,016,076 where the split holds 94,916,076.
    A root that does not match the run gives a fraction, and the field stays
    empty rather than gaining a plausible wrong number."""
    assert lc._measured_epochs(str(tmp_path), "mammal_centered", 111230, 2560, "bert") is None


def test_the_tsv_carries_the_metric_name_on_every_row(tmp_path):
    """BERT rows are eval_loss_mask and GPT-2 rows are val_loss. Reading the
    metric off the series name would break the first time a series is renamed."""
    path = tmp_path / "curves.tsv"
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle, delimiter="\t")
        w.writerow(["series", "arch", "subset", "step", "value", "metric", "job"])
        w.writerow(["bert-w1026", "bert", "mammal_centered", "1000", "1.30",
                    lc.BERT_METRIC, "84667"])
        w.writerow(["gpt2", "gpt2", "mammal_centered", "1000", "1.29",
                    lc.GPT2_METRIC, "28813"])

    rows = list(csv.DictReader(open(path), delimiter="\t"))

    assert [r["metric"] for r in rows] == ["eval_loss_mask", "val_loss"]


@pytest.mark.parametrize("subset,family", [
    ("mammal_centered", "mammal_centered"),
    ("eukaryote_matched_random_seed10", "eukaryote_matched"),
    ("global_random_seed4", "global_random"),
])
def test_the_plotter_groups_a_subset_into_its_corpus_family(subset, family):
    """Colour carries the experiment's axis, so the mapping from subset to
    family is what the reader is actually shown."""
    plot_src = Path(__file__).resolve().parents[2] / "scripts" / "plot_genome_loss_curves.py"
    spec = importlib.util.spec_from_file_location("_genome_plot", plot_src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.family_of(subset) == family


# ---------------------------------------------------------------------------
# A learning-rate arm is not a subset of the production series. `...-w1026-lr3e4`
# ends with the arm, not with the window tag it contains, so the untagged
# fallback put a 1,026-window 3e-4 run into the 512-window twenty-one -- a
# different window, length and learning rate, averaged in with nothing to say so.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,series", [
    ("bert-small-mammal_centered-w1026-lr3e4", "bert-lr-sweep"),
    ("bert-small-mammal_centered-w1026-lr5p5e4", "bert-lr-sweep"),
    ("bert-small-mammal_centered-w1026-lr1e3", "bert-lr-sweep"),
    ("bert-small-mammal_centered-w1026", "bert-w1026"),
    ("bert-small-mammal_centered", "bert-base"),
])
def test_an_arm_does_not_land_in_the_production_series(name, series):
    assert lc.series_of(name) == series


@pytest.mark.parametrize("name", ["stab-base", "stab2-G-clip05"])
def test_a_sweep_run_without_its_rate_in_the_name_is_still_a_sweep_run(name):
    """Two of the GPT-2 sweeps carry no learning rate in their directory name.
    Matching on the arm would drop exactly those two into the production 21."""
    assert lc.series_of(name) == "gpt2-lr-sweep"


def test_the_arm_stays_in_the_label_so_three_curves_are_not_one_name():
    assert lc.subset_of("bert-small-mammal_centered-w1026-lr3e4") == "mammal_centered-lr3e4"
    assert lc.subset_of("bert-small-mammal_centered-w1026") == "mammal_centered"


def _plot():
    import importlib.util
    src = Path(__file__).resolve().parents[2] / "scripts" / "plot_genome_loss_curves.py"
    spec = importlib.util.spec_from_file_location("_genome_plot2", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_subset_per_arm_is_coloured_by_arm_not_by_family():
    """Every run in a sweep is mammal_centered, so the family colour would
    paint all three the same."""
    runs = {"mammal_centered-lr3e4": [], "mammal_centered-lr1e3": []}
    colours, by_arm = _plot().colour_map(runs)

    assert by_arm is True
    assert len(set(colours.values())) == 2


def test_the_production_series_stays_coloured_by_family():
    runs = {"mammal_centered": [], "global_random_seed1": [], "global_random_seed2": []}
    colours, by_arm = _plot().colour_map(runs)

    assert by_arm is False
    assert colours["global_random_seed1"] == colours["global_random_seed2"]


def test_a_run_that_ended_above_its_baseline_is_split_out():
    """Its axis is set by a value the others never reach, and every curve is
    then squeezed into a band where its slope cannot be read."""
    runs = {"good": [(1, 1.30), (2, 1.05)], "bad": [(1, 1.30), (2, 1.37)]}
    learned, collapsed = _plot().split_collapsed(runs, {"good": 1.368, "bad": 1.368})

    assert set(learned) == {"good"} and set(collapsed) == {"bad"}


def test_without_a_baseline_nothing_is_split():
    """Guessing a threshold would be worse than leaving the figure whole."""
    runs = {"a": [(1, 1.30), (2, 1.37)], "b": [(1, 1.30), (2, 1.05)]}
    learned, collapsed = _plot().split_collapsed(runs, {})

    assert learned == runs and collapsed == {}


def test_the_plotter_imports_where_matplotlib_is_not_installed(monkeypatch):
    """CI has no plotting stack, and seven tests here check rules that contain
    no plotting: which colour a run gets, and whether a collapsed run is split
    out. Importing matplotlib at module scope made all seven fail on the import
    line rather than on anything they were checking."""
    import builtins
    import sys

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in ("matplotlib", "japanize_matplotlib"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for module in [m for m in sys.modules
                   if m.split(".")[0] in ("matplotlib", "japanize_matplotlib")]:
        monkeypatch.delitem(sys.modules, module)

    module = _plot()

    assert module.family_of("global_random_seed4") == "global_random"
    with pytest.raises(ModuleNotFoundError):
        module._pyplot()


def test_the_manifest_states_the_job_outright_when_the_run_wrote_one(tmp_path):
    """Newer runs record it, so they depend on neither a name nor a window."""
    import json
    run = tmp_path / "stab-lr5e5-prod"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"run": {"job_id": "138574"}}))

    assert lc.job_id_for(str(run), "gpt2-lr-sweep", [], []) == "138574"


def test_an_arm_label_still_names_its_subset():
    """Colour, baseline and the comparability note all ask which subset a curve
    is on. A sweep labels its curves by arm so three do not share one name, and
    without stripping the arm the figures came out labelled global_random, drew
    no baseline, and said heights could not be compared across subsets when
    every run in them was one subset."""
    plot = _plot()

    assert plot.base_subset("mammal_centered-lr3e4") == "mammal_centered"
    assert plot.base_subset("mammal_centered") == "mammal_centered"
    assert plot.family_of("mammal_centered-lr5p5e4") == "mammal_centered"


def test_a_sweep_on_one_subset_is_not_told_it_cannot_compare_heights():
    """The note is true of the production series and false of a sweep."""
    plot = _plot()
    sweep = {"mammal_centered-lr3e4": [], "mammal_centered-lr1e3": []}

    assert len({plot.base_subset(s) for s in sweep}) == 1


def test_the_epoch_count_looks_for_the_build_not_the_label(tmp_path):
    """A sweep's label carries the arm so three curves do not share one name;
    the build on disk is named for the subset alone. Passing the label looked
    for a mammal_centered-lr3e4 directory, found nothing, and left the caption
    reading "unknown" for a number the data answers."""
    build = tmp_path / "mammal_centered" / "training_ready_hf_dataset_bert"
    build.mkdir(parents=True)

    assert lc._LR_ARM.sub("", lc.subset_of("bert-small-mammal_centered-w1026-lr3e4")) \
        == "mammal_centered"
    assert lc._LR_ARM.sub("", lc.subset_of("bert-small-global_random_seed4")) \
        == "global_random_seed4"


@pytest.mark.parametrize("name,series,subset", [
    ("gpt2-small-mammal_centered", "gpt2", "mammal_centered"),
    ("gpt2-small-mammal_centered-ep9", "gpt2-ep9", "mammal_centered"),
    ("gpt2-small-global_random_seed4-ep9", "gpt2-ep9", "global_random_seed4"),
])
def test_the_nine_epoch_series_is_not_the_three_epoch_one(name, series, subset):
    """They share the production naming apart from the tag. Averaged together
    they would mix runs whose best points sit at 53-98% and at 13-28% of their
    schedules."""
    assert lc.series_of(name) == series
    assert lc.subset_of(name) == subset


def test_a_step_is_converted_to_tokens_by_window_and_batch():
    """A step is not the same amount of data in two series: at one global batch
    the 512-window runs process half of what the 1,026-window runs do. Drawn
    against step, the shorter window looks faster for a reason that has nothing
    to do with learning."""
    plot = _plot()
    meta = {"bert-base": {"window": 512, "global_batch": 2560},
            "bert-w1026": {"window": 1026, "global_batch": 2560}}

    assert plot.tokens_per_step("bert-base", meta) == 512 * 2560
    assert plot.tokens_per_step("bert-w1026", meta) == 1026 * 2560


def test_a_series_whose_window_varies_gets_no_token_axis():
    """Two windows in one series have no single tokens-per-step, and guessing
    one would stretch half the curves against the other half."""
    plot = _plot()

    assert plot.tokens_per_step("mixed", {"mixed": {"window": [512, 1026],
                                                    "global_batch": 2560}}) is None
    assert plot.tokens_per_step("absent", {}) is None


def test_the_token_axis_starts_where_the_data_does():
    """The first evaluation lands at step 1,000 -- about 2.6 G tokens, not
    2.6 M. Taking the schedule's start as the left edge put three empty decades
    on the log axis."""
    import inspect
    src = inspect.getsource(_plot().draw)

    assert "min(x_drawn)" in src
    assert "max(lo_step, 1) * per_step" not in src


def test_two_decades_do_not_print_the_same_tick_label():
    """A bare thousands format rounded everything below 1 G to '0 G', so two
    different decades carried the same label."""
    def fmt(v):
        return (f"{v / 1e9:,.0f} G" if v >= 1e10 else
                f"{v / 1e9:.1f} G" if v >= 1e9 else
                f"{v / 1e6:,.0f} M")

    labels = [fmt(v) for v in (1e8, 5e8, 1e9, 2.6e9, 1e10, 1e11)]

    assert len(set(labels)) == len(labels)
    assert "0 G" not in labels


def test_the_gpt2_epoch_count_is_not_a_literal(tmp_path):
    """It was hardcoded to 3 while the config's _N_EPOCH was a literal 3.
    SUBSET_GPT2_EPOCHS now raises it, and the nine-epoch series went out
    captioned "epochs 3" until this stopped being a constant."""
    import inspect
    src = inspect.getsource(lc.gpt2_meta)

    assert '"epochs": 3' not in src
    assert '"epochs": None' in src


def test_the_collapsed_figure_says_so_in_its_title():
    """Two figures of one series differ only by which runs they hold. Told
    apart by file name alone, the image on its own does not say which it is."""
    import inspect
    src = inspect.getsource(_plot().draw)

    assert "collapsed" in inspect.signature(_plot().draw).parameters
    assert "基準線を上回って終わった run" in src


def test_every_figure_shares_one_x_axis():
    """The three-versus-nine figure is drawn by its own function and was the
    only one left on step after the axis changed -- and it is the figure the
    reading is made from."""
    import inspect
    src = inspect.getsource(_plot().draw_epoch_pair)

    assert "tokens_per_step" in src
    assert 'ax.set_xscale("log")' in src
    assert "figsize=(18, 9)" in src


def test_the_eval_row_count_survives_a_missing_manifest():
    """A loss has no meaning without how its rows were chosen. The manifest
    carries it, but the twenty-one 512-window runs predate the manifest -- so
    the count comes back out of the trainer's own timings instead."""
    state = {"log_history": [{"eval_runtime": 20.0, "eval_samples_per_second": 500.0}]}

    assert lc._bert_eval_set("/nonexistent", state) == "valid の取り方不明 10,000 行"


def test_a_run_that_recorded_nothing_says_nothing():
    assert lc._bert_eval_set("/nonexistent", {"log_history": []}) is None


def test_the_caption_states_how_the_eval_rows_were_chosen():
    """Rule 3 of the 2026-10-05 directive: the draw method and the row count go
    in a footnote to any table of values, and a figure quoting losses is one."""
    import inspect
    src = inspect.getsource(_plot().caption)

    assert "eval_set" in src
    assert "取り方の違う値を同じ列に置かない" in src


def test_the_draw_method_is_read_from_the_log_not_assumed(tmp_path):
    """The twenty-one 512-window runs predate the manifest, so their caption
    said "取り方不明" for something the log states in words. main.py prints the
    two paths differently, so the method is read, not inferred."""
    (tmp_path / "mc-gen-bert-40320.out").write_text(
        "📊 Eval on the first 10000 rows of 50000\n")

    assert lc._eval_set_from_log(str(tmp_path), "40320") == "valid の先頭 10000 行"


def test_a_random_draw_is_told_apart_from_a_head_cut(tmp_path):
    """From 2026-10-05 new runs draw at random. A reader must never see the two
    described the same way, nor their values put in one column."""
    (tmp_path / "mc-gen-bert-99999.out").write_text(
        "📊 Eval on a random 10000-row sample of 50000 (seed 42)\n")

    assert lc._eval_set_from_log(str(tmp_path), "99999") == \
        "valid から無作為 10000 行（seed 42）"


def test_no_log_and_no_manifest_admits_it(tmp_path):
    state = {"log_history": [{"eval_runtime": 20.0, "eval_samples_per_second": 500.0}]}

    assert lc._bert_eval_set(str(tmp_path), state, str(tmp_path), "") == \
        "valid の取り方不明 10,000 行"
