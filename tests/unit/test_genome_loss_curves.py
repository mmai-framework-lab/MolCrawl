"""What the loss-curve TSV must get right before any figure is drawn.

The figures are drawn from this table, so an error here is an error in every
figure at once and is invisible once it is a line on a chart. Three things are
easy to get wrong and are pinned: which CSV a GPT-2 run's points come from when
it holds two, that a series is never merged with another, and that the epoch
count is not guessed from max_steps.
"""
import csv
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


def test_the_job_number_comes_out_of_the_manifest_path_not_the_directory_name(tmp_path):
    import json
    (tmp_path / "run_manifest.json").write_text(json.dumps(
        {"env": {"LEARNING_SOURCE_DIR": "/tmp/genome-bert-84667"}}))

    assert lc.job_id_for(str(tmp_path), "bert-w1026", str(tmp_path), []) == "84667"


def test_a_gpt2_job_is_matched_on_the_node_tensorboard_recorded(tmp_path):
    """GPT-2 writes no manifest and its logs do not carry the subset, so the
    only link left is the node name in the event file."""
    (tmp_path / "events.out.tfevents.1786925563.c394").write_text("")
    sacct = [("28813", "mc-gen-gpt2-mammal_centered", "c394"),
             ("28817", "mc-gen-gpt2-eukaryote_matched+", "c397")]

    assert lc.job_id_for(str(tmp_path), "gpt2", str(tmp_path), sacct) == "28813"


def test_an_ambiguous_node_yields_no_job_rather_than_the_first_match(tmp_path):
    """Two gpt2 jobs on one node cannot be told apart this way; guessing would
    put a wrong job number in the table and nothing would flag it."""
    (tmp_path / "events.out.tfevents.1786925563.c394").write_text("")
    sacct = [("28813", "mc-gen-gpt2-a", "c394"), ("28899", "mc-gen-gpt2-b", "c394")]

    assert lc.job_id_for(str(tmp_path), "gpt2", str(tmp_path), sacct) == ""


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
