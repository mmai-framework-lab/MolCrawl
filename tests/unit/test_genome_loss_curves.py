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
