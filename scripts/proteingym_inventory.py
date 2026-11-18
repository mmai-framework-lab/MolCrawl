"""Build the ProteinGym target inventories and run the step-3 consistency checks.

Issue #388, M1 (second half). For each of the six official benchmark categories this
walks the data CSVs, cross-checks them against the reference CSV, and verifies the
directive's step-3 conditions:

  - reference <-> data correspondence (DMS_id / DMS_filename / target_seq)
  - required columns present (mutated_sequence, DMS_score, DMS_score_bin; subs add mutant)
  - applying the mutant string to the reference (or per-row) sequence reproduces
    mutated_sequence  (substitutions / clinical substitutions; sampled)
  - DMS_score finite; binary label in {0,1} (or Benign/Pathogenic for clinical subs)
  - CV fold columns present and well-formed; cv rows reconcile with the DMS file
  - join key (mutant for substitutions, mutated_sequence for indels) unique per assay

It writes one inventory TSV per category under <base>/inventory and prints a summary.
Exit status is non-zero if any hard problem is found (missing columns, non-finite
scores, bad labels, non-unique keys, or a reference/data mismatch). Sequence-apply
mismatches are reported per assay and counted but, per the directive, recorded rather
than silently fixed; they fail the run so they are not ignored.

Run (from the main checkout, where the data lives):
    python scripts/proteingym_inventory.py [<official_base_dir>]
Default base: learning_source_20260730_protein_uniref50/eval/proteingym/official
"""
from __future__ import annotations

import glob
import math
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO / "learning_source_20260730_protein_uniref50/eval/proteingym/official"

SAMPLE_ROWS_PER_ASSAY = 50  # rows per assay for the mutant-apply check


def apply_mutant(seq: str, mutant: str) -> str | None:
    """Apply a ProteinGym mutant string (e.g. 'A24G' or 'A24G:L50P') to seq (1-indexed).

    Returns the mutated sequence, or None if a wild-type residue does not match (which is
    itself a finding -- a mismatch between the reference sequence and the mutation table).
    """
    chars = list(seq)
    for tok in str(mutant).split(":"):
        tok = tok.strip()
        if not tok:
            continue
        wt, mut = tok[0], tok[-1]
        try:
            pos = int(tok[1:-1])
        except ValueError:
            return None
        if pos < 1 or pos > len(chars) or chars[pos - 1] != wt:
            return None
        chars[pos - 1] = mut
    return "".join(chars)


def _finite(series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    return bool(s.notna().all()) and bool((~s.apply(lambda v: math.isinf(v))).all())


def check_dms(base: Path, data_name: str, ref_name: str, is_indel: bool, out) -> list[str]:
    """Zero-shot DMS (substitutions or indels)."""
    data_dir = base / "data" / data_name
    ref = pd.read_csv(base / "refs" / ref_name)
    ref_by_id = {r["DMS_id"]: r for _, r in ref.iterrows()}
    problems: list[str] = []
    rows = []
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    key = "mutated_sequence" if is_indel else "mutant"
    need = ["mutated_sequence", "DMS_score", "DMS_score_bin"] + ([] if is_indel else ["mutant"])
    seq_mismatch_total = 0
    for f in files:
        dms_id = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        issues = []
        miss = [c for c in need if c not in d.columns]
        if miss:
            issues.append(f"missing_cols={miss}")
            problems.append(f"{data_name}/{dms_id}: missing {miss}")
        if "DMS_score" in d.columns and not _finite(d["DMS_score"]):
            issues.append("DMS_score_nonfinite")
            problems.append(f"{data_name}/{dms_id}: non-finite DMS_score")
        if "DMS_score_bin" in d.columns:
            bad = set(pd.to_numeric(d["DMS_score_bin"], errors="coerce").dropna().unique()) - {0, 1}
            if bad:
                issues.append(f"bin_not01={sorted(bad)}")
                problems.append(f"{data_name}/{dms_id}: DMS_score_bin not 0/1 {sorted(bad)}")
        n_key = d[key].nunique() if key in d.columns else 0
        if key in d.columns and n_key != len(d):
            issues.append(f"dup_key({len(d) - n_key})")
            problems.append(f"{data_name}/{dms_id}: {len(d) - n_key} duplicate {key}")
        ref_row = ref_by_id.get(dms_id)
        if ref_row is None:
            issues.append("no_ref_row")
            problems.append(f"{data_name}/{dms_id}: no reference row")
        # sequence-apply check (substitutions only; indels have no 1:1 position mapping)
        sm = 0
        if not is_indel and ref_row is not None and "mutant" in d.columns:
            tgt = str(ref_row["target_seq"])
            samp = d.sample(min(SAMPLE_ROWS_PER_ASSAY, len(d)), random_state=42)
            for _, rr in samp.iterrows():
                got = apply_mutant(tgt, rr["mutant"])
                if got is None or got != str(rr["mutated_sequence"]):
                    sm += 1
            if sm:
                issues.append(f"seq_mismatch={sm}/{len(samp)}")
                problems.append(f"{data_name}/{dms_id}: {sm}/{len(samp)} mutant->seq mismatch")
        seq_mismatch_total += sm
        rows.append((dms_id, os.path.basename(f), len(d), n_key,
                     int(ref_row["seq_len"]) if ref_row is not None and "seq_len" in ref else "",
                     ";".join(issues)))
    _write_tsv(out, data_name, ["id", "file", "n_rows", "n_unique_key", "seq_len", "issues"], rows)
    print(f"  {data_name}: {len(files)} assays, ref rows={len(ref)}, "
          f"seq_mismatch_rows={seq_mismatch_total}, problems={len([p for p in problems if data_name in p])}")
    return problems


def check_cv(base: Path, data_name: str, dms_name: str, folds: list[str], out) -> list[str]:
    """Supervised CV folds; reconcile against the matching DMS file."""
    data_dir = base / "data" / data_name
    dms_dir = base / "data" / dms_name
    problems: list[str] = []
    rows = []
    for f in sorted(glob.glob(str(data_dir / "*.csv"))):
        dms_id = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        issues = []
        miss = [c for c in folds if c not in d.columns]
        if miss:
            issues.append(f"missing_folds={miss}")
            problems.append(f"{data_name}/{dms_id}: missing fold cols {miss}")
        fold_desc = []
        for fc in folds:
            if fc in d.columns:
                vals = sorted(pd.to_numeric(d[fc], errors="coerce").dropna().astype(int).unique())
                fold_desc.append(f"{fc}:{len(vals)}")
                if pd.to_numeric(d[fc], errors="coerce").isna().any():
                    issues.append(f"{fc}_has_nan")
                    problems.append(f"{data_name}/{dms_id}: {fc} has rows with no fold")
        # The CV file holds the single mutants only, a SUBSET of the DMS file (not the same
        # count). Verify the mutants reconcile: every CV mutant must exist in the DMS file.
        dms_f = dms_dir / os.path.basename(f)
        if dms_f.exists():
            nd_full = pd.read_csv(dms_f)
            nd_full.columns = [c.strip() for c in nd_full.columns]
            if "mutant" in d.columns and "mutant" in nd_full.columns:
                missing_in_dms = set(d["mutant"]) - set(nd_full["mutant"])
                if missing_in_dms:
                    issues.append(f"cv_not_in_dms({len(missing_in_dms)})")
                    problems.append(f"{data_name}/{dms_id}: {len(missing_in_dms)} CV mutants absent from DMS")
            fold_desc.append(f"dms_rows={len(nd_full)}")
        else:
            issues.append("no_matching_dms_file")
            problems.append(f"{data_name}/{dms_id}: no matching DMS file")
        rows.append((dms_id, os.path.basename(f), len(d), ",".join(fold_desc), ";".join(issues)))
    _write_tsv(out, data_name, ["id", "file", "n_rows", "folds", "issues"], rows)
    print(f"  {data_name}: {len(rows)} assays, folds={folds}, "
          f"problems={len([p for p in problems if data_name in p])}")
    return problems


def check_clinical(base: Path, data_name: str, label_col: str, subs: bool, out) -> list[str]:
    """Clinical zero-shot (substitutions use Benign/Pathogenic; indels use 0/1)."""
    data_dir = base / "data" / data_name
    problems: list[str] = []
    rows = []
    seq_mismatch_total = 0
    for f in sorted(glob.glob(str(data_dir / "*.csv"))):
        name = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        issues = []
        if label_col not in d.columns:
            issues.append(f"missing_label={label_col}")
            problems.append(f"{data_name}/{name}: missing {label_col}")
        else:
            vals = set(d[label_col].dropna().unique())
            # Raw clinical labels are the strings Benign/Pathogenic for both subs and indels;
            # some indel files already carry numeric 0/1. prepare-clinical converts to 0/1
            # downstream, so accept either form here.
            allowed = {"Benign", "Pathogenic", 0, 1, "0", "1"}
            if not vals <= allowed:
                issues.append(f"label_vals={sorted(map(str, vals))}")
                problems.append(f"{data_name}/{name}: label not in {allowed}: {sorted(map(str, vals))}")
        if "mutated_sequence" not in d.columns:
            issues.append("missing_mutated_sequence")
            problems.append(f"{data_name}/{name}: missing mutated_sequence")
        # subs: per-row protein_sequence + mutant lets us verify the apply directly
        sm = 0
        if subs and {"protein_sequence", "mutant", "mutated_sequence"} <= set(d.columns):
            samp = d.sample(min(SAMPLE_ROWS_PER_ASSAY, len(d)), random_state=42)
            for _, rr in samp.iterrows():
                got = apply_mutant(str(rr["protein_sequence"]), rr["mutant"])
                if got is None or got != str(rr["mutated_sequence"]):
                    sm += 1
            if sm:
                issues.append(f"seq_mismatch={sm}/{len(samp)}")
                problems.append(f"{data_name}/{name}: {sm}/{len(samp)} mutant->seq mismatch")
        seq_mismatch_total += sm
        rows.append((name, os.path.basename(f), len(d), ";".join(issues)))
    _write_tsv(out, data_name, ["id", "file", "n_rows", "issues"], rows)
    print(f"  {data_name}: {len(rows)} proteins, seq_mismatch_rows={seq_mismatch_total}, "
          f"problems={len([p for p in problems if data_name in p])}")
    return problems


def _write_tsv(out: Path, name: str, header: list[str], rows) -> None:
    p = out / f"inventory-{name}.tsv"
    with open(p, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BASE
    if not (base / "data").is_dir():
        sys.exit(f"no data under {base} -- run workflows/data/proteingym-official-setup.sh first")
    out = base / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    print(f"ProteinGym inventory + step-3 checks  (base={base})")

    problems: list[str] = []
    problems += check_dms(base, "DMS_ProteinGym_substitutions", "DMS_substitutions.csv", False, out)
    problems += check_dms(base, "DMS_ProteinGym_indels", "DMS_indels.csv", True, out)
    problems += check_cv(base, "cv_folds_singles_substitutions", "DMS_ProteinGym_substitutions",
                         ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"], out)
    problems += check_cv(base, "cv_folds_indels", "DMS_ProteinGym_indels", ["fold_random_5"], out)
    problems += check_clinical(base, "clinical_ProteinGym_substitutions", "DMS_bin_score", True, out)
    problems += check_clinical(base, "clinical_ProteinGym_indels", "DMS_score_bin", False, out)

    print(f"\ninventory TSVs -> {out}")
    if problems:
        print(f"\n{len(problems)} problem(s) found (first 25):")
        for p in problems[:25]:
            print(f"  - {p}")
        sys.exit(1)
    print("\nall step-3 checks passed.")


if __name__ == "__main__":
    main()
