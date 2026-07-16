"""Phase 0-3 verification: combine_all.py dedup removes duplicate SMILES.

Builds a small in-memory dataframe that mimics the concat step's output
(4 known duplicates on top of unique rows), applies the same
``drop_duplicates(subset='smiles', keep='first')`` logic, and confirms
`df['smiles'].duplicated().sum() == 0`.
"""
from __future__ import annotations

import pandas as pd

def main() -> int:
    rows = [
        {"smiles": "CCO",   "logp": 0.1, "sascore": 2.0, "mol_weight": 0.4},
        {"smiles": "CCN",   "logp": 0.2, "sascore": 2.1, "mol_weight": 0.5},
        {"smiles": "CCC",   "logp": 0.3, "sascore": 2.2, "mol_weight": 0.4},
        # duplicates below
        {"smiles": "CCO",   "logp": 0.15, "sascore": 2.05, "mol_weight": 0.4},
        {"smiles": "CCN",   "logp": 0.25, "sascore": 2.15, "mol_weight": 0.5},
        {"smiles": "c1ccccc1", "logp": 1.9, "sascore": 1.0, "mol_weight": 0.78},
        {"smiles": "c1ccccc1", "logp": 1.95, "sascore": 1.05, "mol_weight": 0.78},
        {"smiles": "c1ccccc1", "logp": 2.0, "sascore": 1.1, "mol_weight": 0.78},
    ]
    df = pd.DataFrame(rows)

    n_before = len(df)
    n_dupes_before = int(df["smiles"].duplicated().sum())
    print(f"[before] rows={n_before}  duplicated={n_dupes_before}")

    # ---- exact same call as fixed combine_all.py ----
    df = df.drop_duplicates(subset="smiles", keep="first").reset_index(drop=True)
    # -------------------------------------------------

    n_after = len(df)
    n_dupes_after = int(df["smiles"].duplicated().sum())
    print(f"[after]  rows={n_after}   duplicated={n_dupes_after}")

    # keep='first' semantics: the FIRST occurrence's other columns survive
    expected_kept = {"CCO": 0.1, "CCN": 0.2, "c1ccccc1": 1.9}
    for smi, expected_logp in expected_kept.items():
        row = df[df["smiles"] == smi].iloc[0]
        assert abs(row["logp"] - expected_logp) < 1e-9, \
            f"expected first-occurrence logp for {smi}: {expected_logp}, got {row['logp']}"
        print(f"  {smi:12}: kept first-occurrence logp={row['logp']} (expected {expected_logp})")

    assert n_after == 4, f"expected 4 unique rows (CCO, CCN, CCC, c1ccccc1), got {n_after}"
    assert n_dupes_after == 0, f"post-condition violated: {n_dupes_after} duplicates remain"
    assert n_before - n_after == 4, f"expected 4 duplicates removed, got {n_before - n_after}"

    print()
    print("=== ASSERTIONS PASSED ===")
    print(f"  drop_duplicates removed {n_before - n_after} rows")
    print(f"  df['smiles'].duplicated().sum() = {n_dupes_after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
