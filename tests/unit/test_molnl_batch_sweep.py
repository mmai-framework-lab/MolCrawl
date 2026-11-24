"""The molecule_nat_lang batch-size grid: the rules it is derived from, and its configs.

Two things are pinned here.

The derivation. ``scripts/molnl_batch_sweep_arms.py`` turns the 2026-11-20 directive's
rules into 68 arms and compares the result against the directive's own printed tables;
this asserts that comparison comes out clean, and adds the invariants the directive
states in words rather than in a table -- that one GPU's block count can build the
grid's smallest batch, that the three numbers multiply out to it, and that the
intervals fall on whole steps at every batch.

The configs. They are generated, so what is worth testing is not their text but that
each one still carries its own row of the inventory. The reading is by ast, without
importing the repository, so this runs anywhere; the launcher-level check that executes
each config lives in scripts/molnl_batch_sweep_configs.py --check and runs on a compute
node, where the corpus and the tokenizer are reachable.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "molcrawl" / "tasks" / "pretrain" / "configs" / "molecule_nat_lang"


def _load(name):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def arms_module():
    return _load("molnl_batch_sweep_arms")


#: The generator writes this into every config; see scripts/molnl_batch_sweep_configs.py.
SET_MARKER = "# §4 micro-batch set:"


def _declared_set():
    """Which block-count set the configs on disk were generated from.

    Without this the comparison below would derive one set and read configs written
    from another, which is what happened when the §4 measurement replaced the set the
    configs had been written with: 45 tests passed and the one that compares the two
    failed on a difference that was not a defect.
    """
    for path in sorted(CONFIG_DIR.glob("*_b*_lr*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(SET_MARKER):
                return line[len(SET_MARKER):].strip()
    return "current"


@pytest.fixture(scope="module")
def micro_set():
    return _declared_set()


@pytest.fixture(scope="module")
def rows(arms_module, micro_set):
    return arms_module.arms(micro_set)


def test_sixty_eight_arms(arms_module, rows):
    assert len(rows) == sum(arms_module.EXPECTED_ARMS.values()) == 68
    for arch, want in arms_module.EXPECTED_ARMS.items():
        assert sum(1 for r in rows if r["arch"] == arch) == want


def test_derivation_agrees_with_the_directive(arms_module, rows, micro_set, capsys):
    """§1-§4's printed tables against the rules they follow from."""
    problems = arms_module.check(rows, micro_set)
    capsys.readouterr()
    assert problems == 0


def test_every_registered_set_derives_cleanly(arms_module, capsys):
    """Not only the one the configs happen to carry."""
    for name in sorted(arms_module.MICRO_BATCH):
        problems = arms_module.check(arms_module.arms(name), name)
        capsys.readouterr()
        assert problems == 0, name


def test_the_configs_name_the_set_they_were_written_from(rows):
    """The marker is what the comparison above relies on, so it has to be there."""
    for row, path in _config_paths(rows):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert f"{SET_MARKER} {row['micro_batch_set']}" in text, path.name


def test_the_three_numbers_multiply_out_to_the_batch(rows):
    for row in rows:
        micro = row["micro_batch"]
        accum = row["grad_accum"]
        world = row["world_size"]
        assert micro * accum * world == row["batch"] == row["expected_global_batch"], row


def test_one_gpu_can_build_the_smallest_batch(arms_module, rows):
    """§4: one GPU's block count times 4 GPUs must divide 256."""
    for row in rows:
        assert row["micro_batch"] in arms_module.MICRO_CANDIDATES, row
        assert 256 % (row["micro_batch"] * 4) == 0, row


def test_one_gpu_block_count_is_held_across_the_batches(rows):
    """§4: it is chosen per size and does not move between the five batches."""
    per_size = {}
    for row in rows:
        per_size.setdefault((row["arch"], row["size"]), set()).add(row["micro_batch"])
    for key, values in per_size.items():
        assert len(values) == 1, (key, values)


def test_the_budget_is_the_same_for_every_arm_of_a_framework(arms_module, rows):
    for arch, blocks in arms_module.BUDGET_BLOCKS.items():
        sel = [r for r in rows if r["arch"] == arch]
        assert {r["blocks"] for r in sel} == {blocks}
        assert {r["tokens"] for r in sel} == {blocks * arms_module.SEQUENCE_LENGTH}
        # §2: the step count is the budget divided by the batch, exactly.
        for row in sel:
            assert row["steps"] * row["batch"] == blocks, row


def test_intervals_land_on_whole_steps_and_on_the_same_blocks(arms_module, rows):
    for row in rows:
        every = arms_module.EVAL_EVERY_BLOCKS[row["arch"]]
        assert row["eval_interval"] * row["batch"] == every, row
        assert row["save_interval"] % row["eval_interval"] == 0, row
        assert row["steps"] % row["eval_interval"] == 0, row
        assert row["eval_points"] * every == row["blocks"], row


def test_warmup_is_the_frameworks_own_fraction(arms_module, rows):
    for row in rows:
        fraction = arms_module.WARMUP_FRACTION[row["arch"]]
        assert row["warmup_steps"] == int(fraction * row["steps"]), row
        # A warmup as long as the run would never reach the peak.
        assert row["warmup_steps"] < row["steps"], row


def test_gpt2_floor_and_decay_length(arms_module, rows):
    for row in rows:
        if row["arch"] != "gpt2":
            assert row["min_lr"] is None and row["lr_decay_iters"] is None, row
            continue
        assert row["lr_decay_iters"] == row["steps"], row
        ratio = float(row["lr"]) / float(row["min_lr"])
        assert abs(ratio - arms_module.GPT2_MIN_LR_DIVISOR) < 0.02, row


def test_every_arm_has_its_own_config_and_directory(rows):
    assert len({r["config"] for r in rows}) == len(rows)
    assert len({r["out_dir"] for r in rows}) == len(rows)
    for row in rows:
        stem = Path(row["config"]).stem
        assert stem == f"{row['arch']}_{row['size']}_b{row['batch']}_lr{row['lr_slug']}"
        # bert-grid.sbatch refuses a model_path whose basename is not named for a rate,
        # which is what stops two arms of a size sharing one directory.
        assert "-lr" in row["out_dir"]


def test_bert_evaluation_is_held_across_the_grid(arms_module, rows):
    bert = [r for r in rows if r["arch"] == "bert"]
    assert {r["eval_batch"] for r in bert} == {arms_module.BERT_EVAL_BATCH}
    assert {r["eval_blocks"] for r in bert} == {arms_module.BERT_EVAL_BLOCKS}
    assert {r["world_size"] for r in bert} == {4}
    assert {r["judge_metric"] for r in bert} == {"eval_loss_mask"}
    assert {r["judge_metric"] for r in rows if r["arch"] == "gpt2"} == {"val_loss"}


# ---------------------------------------------------------------------------
# The 68 generated configs
# ---------------------------------------------------------------------------

def _declared(path):
    """Top-level literal assignments, by ast, without importing anything."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except ValueError:
            out[target.id] = None  # a call or a name: read from the text instead
    return out


def _config_paths(rows):
    return [(row, CONFIG_DIR / Path(row["config"]).name) for row in rows]


def test_every_arm_has_its_config_on_disk(rows):
    missing = [row["config"] for row, path in _config_paths(rows) if not path.exists()]
    assert not missing, f"{len(missing)} of {len(rows)} configs are not written: {missing[:5]}"


def test_configs_carry_their_inventory_row(rows):
    keys = {
        "bert": [("steps", "max_steps"), ("warmup_steps", "warmup_steps"),
                 ("eval_interval", "log_interval"), ("save_interval", "save_steps"),
                 ("micro_batch", "batch_size"),
                 ("grad_accum", "gradient_accumulation_steps"),
                 ("expected_global_batch", "expected_global_batch"),
                 ("eval_batch", "per_device_eval_batch_size")],
        "gpt2": [("steps", "max_iters"), ("warmup_steps", "warmup_iters"),
                 ("eval_interval", "eval_interval"),
                 ("save_interval", "save_checkpoint_steps"),
                 ("lr_decay_iters", "lr_decay_iters"),
                 ("micro_batch", "batch_size"),
                 ("grad_accum", "gradient_accumulation_steps"),
                 ("expected_global_batch", "expected_global_batch")],
    }
    for row, path in _config_paths(rows):
        if not path.exists():
            pytest.skip("configs are not written yet")
        values = _declared(path)
        for column, key in keys[row["arch"]]:
            assert values.get(key) == int(row[column]), (path.name, key)
        assert values.get("learning_rate") == pytest.approx(float(row["lr"]), rel=1e-12), path.name
        assert values.get("seed") == 42, path.name
        assert values.get("early_stopping") is False, path.name
        if row["arch"] == "gpt2":
            assert values.get("min_lr") == pytest.approx(float(row["min_lr"]), rel=1e-12), path.name
            assert values.get("init_from") == "scratch", path.name
            assert values.get("deterministic_val_eval") is True, path.name
        else:
            assert values.get("fixed_eval_mask") is True, path.name
            assert values.get("eval_mask_seed") == 42, path.name
            for key in ("judge_on", "metric_for_best_model", "save_on_improve_metric",
                        "checkpoint_metric"):
                assert values.get(key) == "eval_loss_mask", (path.name, key)
            assert values.get("save_on_improve") is True, path.name


def test_configs_do_not_declare_an_eval_subset(rows):
    """molnl scores its whole 1,631-block validation split; see §6 in the generator."""
    for row, path in _config_paths(rows):
        if not path.exists() or row["arch"] != "bert":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not line.startswith(("eval_subset_random", "eval_subset_seed")), path.name


def test_configs_name_their_own_output_directory(rows):
    for row, path in _config_paths(rows):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        suffix = f'-b{row["batch"]}-lr{row["lr_slug"]}"'
        assert suffix in text, (path.name, suffix)
        assert "get_bert_output_path(" in text or "get_gpt2_output_path(" in text


def test_configs_inherit_their_size_base(rows):
    """Not a style point: the base carries preprocess_function and the vocab guard."""
    for row, path in _config_paths(rows):
        if not path.exists():
            continue
        base = f"{'bert' if row['arch'] == 'bert' else 'gpt2'}_{row['size']}"
        want = (f"from molcrawl.tasks.pretrain.configs.molecule_nat_lang.{base} "
                f"import *")
        assert want in path.read_text(encoding="utf-8"), (path.name, want)


# --- the curve reader (§10, §12) ---------------------------------------------------
#
# The readers themselves need a finished run to read and are checked against one by
# hand; what is pinned here is the collapse rule, which decides a reported column and
# must not depend on a floor constant quoted from the wrong metric or split.


@pytest.fixture(scope="module")
def curves_module():
    return _load("molnl_batch_sweep_curves")


def test_a_descending_arm_has_not_collapsed(curves_module):
    series = [(100, 5.80), (200, 3.61), (300, 1.20), (400, 0.42), (500, 0.06)]
    assert curves_module.collapse_at(series, None) is None


def test_a_non_finite_value_is_a_collapse_wherever_it_sits(curves_module):
    series = [(100, 5.80), (200, float("nan")), (300, 1.20)]
    assert curves_module.collapse_at(series, None) == 200
    series = [(100, 5.80), (200, 1.20), (300, float("inf"))]
    assert curves_module.collapse_at(series, None) == 300


def test_coming_back_to_the_first_point_is_a_collapse(curves_module):
    """The arm descended, then returned to where it had learned nothing."""
    series = [(100, 5.80), (200, 1.20), (300, 0.42), (400, 3.90), (500, 5.90)]
    assert curves_module.collapse_at(series, None) == 500


def test_the_first_points_do_not_trigger_the_rule_themselves(curves_module):
    """Point 1 is >= itself; only points after the minimum are tested."""
    series = [(100, 5.80), (200, 5.79)]
    assert curves_module.collapse_at(series, None) is None


def test_a_given_floor_catches_a_collapse_that_stops_short_of_the_start(curves_module):
    series = [(100, 5.80), (200, 1.20), (300, 0.42), (400, 4.70)]
    assert curves_module.collapse_at(series, None) is None
    assert curves_module.collapse_at(series, 4.6449) == 400


def test_the_floor_needs_its_provenance(curves_module):
    with pytest.raises(SystemExit):
        curves_module.main(["--arms", "x", "--root", "y", "--curves-out", "rt1-a.tsv",
                            "--summary-out", "rt1-b.tsv", "--floor", "4.6449"])


def test_the_curve_and_summary_columns_are_what_the_directive_asks_for(curves_module):
    for column in ("batch", "rows_seen", "tokens_seen", "split", "metric", "value"):
        assert column in curves_module.CURVE_COLUMNS, column
    for column in ("best_value", "best_rows", "best_fraction_of_budget",
                   "final_value", "final_minus_best", "steps_reached",
                   "collapsed", "collapse_rows"):
        assert column in curves_module.SUMMARY_COLUMNS, column
    # R12: a value column comes with a metric column, and the summary has neither.
    assert "value" not in curves_module.SUMMARY_COLUMNS


def test_bert_contributes_the_metrics_section_7_asks_to_record(curves_module):
    names = {name for _key, _split, name in curves_module.BERT_METRICS}
    assert {"eval_loss_mask", "eval_loss", "eval_loss_copy", "eval_loss_random",
            "train_loss_mask"} == names


# --- the §9 table -------------------------------------------------------------------


@pytest.fixture(scope="module")
def shapes_table_module():
    return _load("molnl_batch_sweep_shapes_table")


def test_a_shape_result_line_with_spaces_in_its_last_value_parses(shapes_table_module,
                                                                  tmp_path):
    """The timing script's `source=` value holds spaces; splitting on them loses it."""
    log = tmp_path / "probe.out"
    log.write_text(
        "SHAPE_RESULT arch=bert label=medium-16x40 per_step_s=7.612 median_s=7.600 "
        "stdev_s=0.110 steps_read=26 accum=40 source=train_runtime minus the "
        "evaluation time, over the steps\n"
        "SHAPE_RESULT arch=gpt2 label=xl-4x640 per_step_s= failed=1\n",
        encoding="utf-8")
    got = shapes_table_module.parse_shape_results([str(log)])
    assert got[("bert", "medium", 16)]["per_step_s"] == "7.612"
    assert got[("bert", "medium", 16)]["source"].startswith("train_runtime minus")
    assert got[("gpt2", "xl", 4)]["failed"] == "1"


def test_todays_shapes_are_read_from_the_derivation(shapes_table_module, arms_module):
    assert shapes_table_module.todays_shapes() == arms_module.MICRO_BATCH["current"]


def test_the_table_refuses_an_inventory_whose_size_has_two_shapes(shapes_table_module,
                                                                  tmp_path):
    header = ("arch\tsize\tbatch\tlr\tlr_multiplier\tmicro_batch\tgrad_accum\t"
              "world_size\texpected_global_batch\tsteps\teval_interval\tsave_interval\n")
    arms = tmp_path / "rt1-two-shapes.tsv"
    arms.write_text(
        header
        + "bert\tsmall\t256\t0.000316\t1.0\t16\t4\t4\t256\t120000\t480\t2400\n"
        + "bert\tsmall\t512\t0.000447\t1.0\t8\t16\t4\t512\t60000\t240\t1200\n",
        encoding="utf-8")
    with pytest.raises(SystemExit):
        shapes_table_module.main(["--arms", str(arms)])


def test_the_output_directory_names_its_architecture(rows):
    """BERT medium and GPT-2 medium scale the same 2,560 rate to the same slug.

    Without the architecture in the name, five pairs of arms carry one name between
    them. The parent directory keeps them apart on disk; the name has to keep them
    apart everywhere else.
    """
    for row in rows:
        assert f"-{row['arch']}-b{row['batch']}-" in row["out_dir"], row["out_dir"]


# --- the §4 choice ------------------------------------------------------------------


@pytest.fixture(scope="module")
def choice_module():
    return _load("molnl_batch_shape_choice")


def _probe_log(tmp_path, lines):
    log = tmp_path / "mc-molnl-bshape-probe.out"
    log.write_text("".join(lines), encoding="utf-8")
    return log


def test_bert_is_ranked_on_its_per_step_time(choice_module, tmp_path):
    log = _probe_log(tmp_path, [
        "SHAPE_RESULT arch=bert label=small-8x80 per_step_s=3.851 steps=50\n",
        "SHAPE_RESULT arch=bert label=small-16x40 per_step_s=2.897 steps=50\n",
        "SHAPE_RESULT arch=bert label=small-64x10 per_step_s=2.517 steps=50\n",
    ])
    chosen, table, problems = choice_module.choose(choice_module.parse([str(log)]))
    assert chosen["bert"]["small"] == 64
    assert any(r.get("chosen") and r["margin_s"] == pytest.approx(0.380)
               for r in table)
    # Only one size of one architecture was measured, so the rest are named.
    assert any("gpt2" in p for p in problems)


def test_gpt2_is_ranked_on_the_median_not_the_mean(choice_module, tmp_path):
    """The mean follows a few iterations that are not the step; the first probe had
    mean 3.472 against median 2.634."""
    log = _probe_log(tmp_path, [
        "SHAPE_RESULT arch=gpt2 label=large-8x320 per_step_s=3.472 median_s=2.634 "
        "stdev_s=2.846 iters_read=35\n",
        "SHAPE_RESULT arch=gpt2 label=large-16x160 per_step_s=3.000 median_s=2.900 "
        "stdev_s=0.100 iters_read=35\n",
    ])
    chosen, _table, _problems = choice_module.choose(choice_module.parse([str(log)]))
    # On the mean, 16 would win; on the median, 8 does.
    assert chosen["gpt2"]["large"] == 8


def test_one_usable_shape_is_not_a_choice(choice_module, tmp_path):
    log = _probe_log(tmp_path, [
        "SHAPE_RESULT arch=bert label=xl-4x160 per_step_s=26.034 steps=20\n",
        "SHAPE_RESULT arch=bert label=xl-8x80 per_step_s= failed=1\n",
    ])
    chosen, _table, problems = choice_module.choose(choice_module.parse([str(log)]))
    assert "xl" not in chosen.get("bert", {})
    assert any("usable shape" in p for p in problems)


def test_a_shape_that_cannot_build_256_is_refused(choice_module, tmp_path):
    log = _probe_log(tmp_path, [
        "SHAPE_RESULT arch=bert label=small-24x? per_step_s=1.000 steps=50\n",
        "SHAPE_RESULT arch=bert label=small-16x40 per_step_s=2.897 steps=50\n",
        "SHAPE_RESULT arch=bert label=small-64x10 per_step_s=2.517 steps=50\n",
    ])
    chosen, _table, problems = choice_module.choose(choice_module.parse([str(log)]))
    assert chosen["bert"]["small"] == 64
    assert any("24" in p for p in problems)


def test_an_incomplete_choice_file_is_not_offered_as_a_set(arms_module):
    """The derivation registers 'measured-1120' only when all eight cells are in it."""
    sets = sorted(arms_module.MICRO_BATCH)
    assert "current" in sets
    if "measured-1120" in sets:
        measured = arms_module.MICRO_BATCH["measured-1120"]
        for arch in ("bert", "gpt2"):
            assert sorted(measured[arch]) == sorted(arms_module.SIZES)
            for micro in measured[arch].values():
                assert micro in arms_module.MICRO_CANDIDATES
                assert 256 % (micro * 4) == 0


def test_the_manifest_reader_takes_the_output_directory_from_the_inventory(rows):
    """One spelling of the output directory, not two.

    The reader composed the name itself from size, batch and rate, which left out the
    architecture that was added to it after BERT medium and GPT-2 medium turned out to
    share a rate. Nothing errored: every arm read as not started, and §11 reported 0
    arms with 0 disagreements, which is indistinguishable from a pass. So the name has
    to come from the inventory the configs were generated from.
    """
    manifests = _load("molnl_batch_sweep_manifests")
    for arm in rows:
        where = manifests.run_dir({k: str(v) for k, v in arm.items()}, "/root")
        assert where == f"/root/molecule_nat_lang/{'bert-output' if arm['arch'] == 'bert' else 'gpt2-output'}/{arm['out_dir']}", arm
        assert f"-{arm['arch']}-" in where, (arm["arch"], where)


def test_a_non_numeric_expectation_stays_text(rows):
    """§7's three selection keys are compared against a metric name, not a number.

    ``expectation`` cast every inventory column through int then float, so the first arm
    with a manifest to read raised on 'eval_loss_mask'. The reader had therefore never
    been run against a real run.
    """
    manifests = _load("molnl_batch_sweep_manifests")
    arm = {k: str(v) for k, v in rows[0].items()}
    assert manifests.expectation("@judge_metric", arm) == arm["judge_metric"]
    assert manifests.expectation("@batch", arm) == int(arm["batch"])
    assert manifests.expectation("@lr", arm) == pytest.approx(float(arm["lr"]))
    assert manifests.same(arm["judge_metric"], arm["judge_metric"])


def test_every_field_the_manifest_row_sets_has_a_column():
    """The writer refuses a row with a field the header does not have (convention R2).

    per_device_eval_batch_size is in neither key table, because it has no manifest path:
    the BERT manifest does not record it and it is read from the config. It was set on
    the row and left out of the header, so the whole table went unwritten after 1,424
    values had been compared.
    """
    manifests = _load("molnl_batch_sweep_manifests")
    for name, keys in (("bert", manifests.BERT_KEYS), ("gpt2", manifests.GPT2_KEYS)):
        for key in keys:
            assert key[0] in manifests.COLUMNS, (name, key[0])
    assert "per_device_eval_batch_size" in manifests.COLUMNS


def test_every_set_a_config_declares_is_in_the_module(arms_module):
    """A config's §4 set has to be derivable from the repository alone.

    The measured set was loaded from scripts/molnl_batch_shape_choice_20261120.json,
    which matches .gitignore's ``*.json`` and so had never been committed. The 68 configs
    name that set, the module only had it where the probes had run, and eighteen tests
    raised KeyError('measured-1120') in CI while passing on one node. The set is a
    literal now; this asserts the configs cannot get ahead of the module again.
    """
    declared = set()
    for path in sorted(CONFIG_DIR.glob("*_b[0-9]*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(SET_MARKER):
                declared.add(line[len(SET_MARKER):].strip())
                break
    assert declared, f"no config under {CONFIG_DIR} carries a {SET_MARKER!r} line"
    missing = sorted(declared - set(arms_module.MICRO_BATCH))
    assert missing == [], missing


def test_the_measured_set_matches_the_probe_output_it_came_from(arms_module):
    """The literal and its provenance file agree, wherever the file is present.

    The file is committed, so this runs everywhere rather than only where the probes
    ran. The module raises on a disagreement at import; this says so in a test as well,
    because the import-time check is easy to read past.
    """
    path = Path(arms_module.CHOICE_FILE)
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    choice = json.loads(path.read_text(encoding="utf-8"))
    typed = arms_module.MICRO_BATCH[choice["set"]]
    for arch in ("bert", "gpt2"):
        for size in arms_module.SIZES:
            assert typed[arch][size] == choice["chosen"][arch][size], (arch, size)
