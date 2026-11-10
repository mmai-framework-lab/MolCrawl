"""What the §3 seed inventory must not get wrong.

1. The two seed columns come from different places in the manifest and must not be
   collapsed: ``sources.seed.value`` is what the config asked for,
   ``seed.seed`` is what transformers seeded with. Six runs disagree.
2. A run with no manifest leaves both of those columns empty. Its recovered
   ``args.seed`` goes in its own column, so no row claims a manifest it has not got.
3. A run whose history cannot be read is still a run. It stays in the list.
4. The file follows the report TSV convention: version line, provenance with
   ``model``/``source``/``generated``, ASCII column names, no BOM, LF endings.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from bert_seed_inventory import (  # noqa: E402
    COLUMNS,
    collect,
    inventory_row,
    provenance,
    row_without_trainer_state,
    write_tsv,
)


def write_manifest(run_dir, config_seed=None, used_seed=None, lr=None,
                   job_id="123456", seed_from="config file", size=None):
    os.makedirs(run_dir, exist_ok=True)
    manifest = {"run": {"job_id": job_id}, "schedule": {}, "sources": {}}
    if lr is not None:
        manifest["schedule"]["learning_rate"] = lr
    if used_seed is not None:
        manifest["seed"] = {"seed": used_seed}
    if config_seed is not None:
        manifest["sources"]["seed"] = {"value": config_seed, "from": seed_from}
    if size is not None:
        manifest["introduced"] = {"model_size": size}
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return manifest


def audit_row(run_dir, modality="rna", run=None, size="medium", lr=3e-4,
              job_id="123456", steps_run=24000):
    return {
        "modality": modality, "arch": "bert",
        "run": run or os.path.basename(run_dir), "run_dir": run_dir,
        "size": size, "learning_rate": lr, "job_id": job_id, "steps_run": steps_run,
    }


def test_the_two_seed_columns_come_from_different_manifest_paths(tmp_path):
    run_dir = str(tmp_path / "protein_sequence-large-lr5e5-s17")
    write_manifest(run_dir, config_seed=17, used_seed=42, lr=5e-5)
    row = inventory_row(audit_row(run_dir, "protein_sequence", size="large"), "tree")
    assert row["config_seed"] == 17
    assert row["manifest_seed"] == 42
    assert row["seed_agrees"] == "no"
    assert row["seed_source"] == "run_manifest.json"
    assert row["args_seed"] == ""


def test_agreement_is_marked_yes_not_left_blank(tmp_path):
    run_dir = str(tmp_path / "rna-medium-lr3e4")
    write_manifest(run_dir, config_seed=42, used_seed=42)
    row = inventory_row(audit_row(run_dir), "tree")
    assert row["seed_agrees"] == "yes"


def test_a_run_without_a_manifest_keeps_both_columns_empty(tmp_path):
    run_dir = str(tmp_path / "compounds-large")
    os.makedirs(run_dir)
    calls = []

    def fake_reader(path):
        calls.append(path)
        return 42, "checkpoint-15000/training_args.bin"

    row = inventory_row(audit_row(run_dir, "compounds", size="large"), "tree",
                        read_args_seed=fake_reader)
    assert row["config_seed"] == ""
    assert row["manifest_seed"] == ""
    assert row["args_seed"] == 42
    assert row["seed_source"] == "checkpoint-15000/training_args.bin"
    assert row["seed_agrees"] == ""
    assert calls == [run_dir]


def test_a_seed_that_cannot_be_established_says_so(tmp_path):
    run_dir = str(tmp_path / "genome-small-aborted")
    os.makedirs(run_dir)
    row = inventory_row(audit_row(run_dir, "genome_sequence", size="small"), "tree",
                        read_args_seed=lambda _d: (None, "no training_args.bin in any checkpoint"))
    assert row["args_seed"] == ""
    assert row["seed_source"] == "no training_args.bin in any checkpoint"


def test_a_run_with_no_readable_history_is_still_listed(tmp_path):
    run_dir = str(tmp_path / "rna-small-lr3e4-crashed")
    write_manifest(run_dir, config_seed=106, used_seed=106, lr=3e-4, size="small")
    # It wrote checkpoints and then lost its history, so only the one note applies.
    entry = {"modality": "rna", "run_dir": run_dir, "checkpoints": [2000, 1000],
             "reason": "no readable trainer_state.json"}
    base = row_without_trainer_state(entry, "rna")
    assert base["size"] == "small"
    assert base["learning_rate"] == 3e-4
    assert base["checkpoints_present"] == [1000, 2000]
    row = inventory_row(base, "tree", note="no readable trainer_state.json")
    assert row["manifest_seed"] == 106
    assert row["note"] == "no readable trainer_state.json"
    assert row["steps_run"] == ""


def test_a_crashed_run_with_no_checkpoint_carries_both_notes(tmp_path):
    run_dir = str(tmp_path / "rna-small-lr3e4-died-early")
    write_manifest(run_dir, config_seed=106, used_seed=106)
    base = row_without_trainer_state(
        {"modality": "rna", "run_dir": run_dir, "checkpoints": []}, "rna")
    row = inventory_row(base, "tree", note="no readable trainer_state.json")
    assert row["note"] == ("no readable trainer_state.json;"
                           " manifest only (no checkpoint)")
    assert row["production"] == "no"


def test_gpt2_runs_are_not_in_this_inventory(tmp_path, monkeypatch):
    bert_dir = str(tmp_path / "rna-medium-lr3e4")
    gpt2_dir = str(tmp_path / "rna-gpt2-medium-lr1p2e3-9ep")
    write_manifest(bert_dir, config_seed=42, used_seed=42)
    write_manifest(gpt2_dir, config_seed=42, used_seed=42)
    rows_found = [audit_row(bert_dir),
                  dict(audit_row(gpt2_dir), arch="gpt2")]
    monkeypatch.setattr("bert_seed_inventory.walk_root",
                        lambda root, modality: (rows_found, []))
    rows = collect([("rna", str(tmp_path))])
    assert [r["run"] for r in rows] == [os.path.basename(bert_dir)]


def test_production_flag_follows_the_audit_rule(tmp_path):
    prod = str(tmp_path / "rna-medium-lr3e4")
    diag = str(tmp_path / "rna-medium-lr3e4-diag")
    nojob = str(tmp_path / "rna-medium-lr3e4-old")
    for d in (prod, diag, nojob):
        write_manifest(d, config_seed=42, used_seed=42)
    assert inventory_row(audit_row(prod), "t")["production"] == "yes"
    assert inventory_row(audit_row(diag), "t")["production"] == "no"
    assert inventory_row(audit_row(nojob, job_id=None), "t")["production"] == "no"


def test_a_run_that_wrote_no_checkpoint_is_not_production(tmp_path):
    """The two throughput probes hold a manifest and nothing else.

    One of their names matches no diagnostic word, so the name-based rule alone
    would call it production.
    """
    run_dir = str(tmp_path / "protein_small_fp32_w4-121575")
    write_manifest(run_dir, config_seed=20260920, used_seed=20260920)
    row = inventory_row(audit_row(run_dir, "protein_sequence", size="small",
                                  steps_run=None), "learning_source_bert_tput")
    assert row["production"] == "no"
    assert "manifest only (no checkpoint)" in row["note"]
    assert row["manifest_seed"] == 20260920


def test_the_file_follows_the_report_tsv_convention(tmp_path):
    run_dir = str(tmp_path / "rna-medium-lr3e4")
    write_manifest(run_dir, config_seed=42, used_seed=42)
    rows = [inventory_row(audit_row(run_dir), "learning_source_rna_runs")]
    out = str(tmp_path / "rt1-bert-seed-inventory.tsv")
    write_tsv(out, rows, provenance([("rna", "/somewhere/learning_source_rna_runs")],
                                    "scripts/bert_seed_inventory.py"))

    raw = open(out, "rb").read()
    assert not raw.startswith(b"\xef\xbb\xbf")        # R5: no BOM
    assert b"\r\n" not in raw                          # R6: LF only
    text = raw.decode("utf-8").splitlines()
    assert text[0] == "#! report-tsv 1"                # R13
    comments = [line for line in text if line.startswith("#")]
    for key in ("# model:", "# source:", "# generated:"):  # R14
        assert any(line.startswith(key) for line in comments)
    header = next(line for line in text if not line.startswith("#"))
    names = header.split("\t")
    assert names == COLUMNS
    assert len(names) == len(set(names))               # R4
    for name in names:                                 # R3: ASCII only
        assert name.replace("_", "").replace("@", "").replace(".", "").isalnum()
    body = [line for line in text if not line.startswith("#")][1:]
    assert all(len(line.split("\t")) == len(names) for line in body)  # R2
    assert "不明" not in "\n".join(comments)            # R15
