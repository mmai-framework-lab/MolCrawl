# Audit archive — v2 CSV incident (2026-07-29)

## Purpose

Preserved artifacts related to the discovery that the entire 21-subset genome
pretraining corpus was built from the v1 CSVs snapshotted at repo commit
`60dc5368` (2026-06-05), while an upstream v2 CSV set was available at
`/lustre/home/kojima-t/data/genome_subsets_v2/subsets_v2/` from 2026-07-10 but
never reached the F2 preparation pipeline.

Kept in-repo (rather than filesystem-only) because these are the primary
evidence for:

1. Reproducibility of the incident (the exact v1 CSV bytes the pipeline
   consumed for BERT × 21 + GPT-2 × 21 training runs on the Octopus cluster
   during 2026-07-14 → 2026-07-30).
2. Point-in-time state of the autopilot coordinator during the 2026-07-21
   → 2026-07-27 retry sequences (seed10 NODE_FAIL, LR-fix pauses, LOST-17
   rerun kick, best-ckpt fix restart).

Related documents (personal working memos, not committed):
- `tmp/docs_tmp_local/yigarashi-issue/20260729-CRITICAL-all21-subsets-mismatch.md`
- `tmp/docs_tmp_local/yigarashi-issue/20260729-v2-csv-provenance-neutral-summary.md`
- `tmp/docs_tmp_local/yigarashi-issue/20260729-seed6-rebuild-csv-discrepancy.md`
- `tmp/docs_tmp_local/20260729-matsubara-technical-recurrence-prevention.md`

## Contents

### v1_csv_snapshot.tar.gz

Archive of the 21 CSV files at `assets/genome_species_list/subsets/*.csv`
as they existed on the `feat/production-finalize/1-5-bert-large-lr-unify-and-subset-training`
branch. Created 2026-05-29 during pre-Phase-2 planning; captures the
`60dc5368`-era CSV state that later became the "v1" side of the v1/v2
mismatch.

| Path in tar | Purpose |
|---|---|
| `subsets/mammal_centered.csv` | mammal-centered subset (28 species) |
| `subsets/eukaryote_matched_random_seed{1..10}.csv` | 10 eukaryote-matched random subsets |
| `subsets/global_random_seed{1..10}.csv` | 10 global random subsets (incl. the anomalous seed6) |

sha256: `31032710980d68e4a541d173b251af2ddab677b994bc73ff46d959e28518c882`
Size: 755,659 bytes (738 KiB)
Origin timestamp: 2026-05-29 15:56 JST

### coordinator_state_bak/ (10 files)

Autopilot coordinator state snapshots taken before each recovery / config
change during the 2026-07-21 → 2026-07-27 window. Each `.bak` was written
manually before the autopilot state was mutated for a specific retry or
config-flip event.

| File | Trigger event |
|---|---|
| `coordinator_state.json.bak_20260721_095127` | pre-seed10-resubmit baseline |
| `coordinator_state.json.bak_20260722_105534_lrfix_pause` | first LR-fix pause |
| `coordinator_state.json.bak_20260722_105738_lrfix_pause2_bertseed10` | second LR-fix pause (BERT seed10) |
| `coordinator_state.json.bak_20260722_112047_seed10_retry3` | seed10 3rd retry |
| `coordinator_state.json.bak_20260723_105404_gpt2_rerun_lr1e-4` | GPT-2 rerun at LR 1e-4 |
| `coordinator_state.json.bak_20260724_091453_gpt2_seed5_retry` | GPT-2 seed5 retry |
| `coordinator_state.json.bak_20260724_094530_bestckpt_fix_restart` | best-ckpt protection fix restart |
| `coordinator_state.json.bak_20260724_164033_lost17_rerun_kick` | LOST-17 rerun kick |
| `coordinator_state.json.bak_20260727_091301_seed8_retry42` | seed8 retry after 2nd NODE_FAIL |

Total size: ~360 KiB (10 files, JSON, each 20-70 KiB)

sha256:
- `08735c5d777d4a2ee54b9d9f9a8d7f51b2b8f536f10db41c869df8e4eb54fc97` bak_20260721_095127
- `60c2f27eda5cc64cfeadbe71abfae29278e3fd91f8ad2c90972174713628c915` bak_20260722_105534_lrfix_pause
- `5c2cb1acdf4d8ea80dcfafaa65c9207d01bf8fef4d75874db4d26198fddd2292` bak_20260722_105738_lrfix_pause2_bertseed10
- `506b0b45ba90c8fbfef78d4afda5c966d88b2739a660f7462782da19f4eae349` bak_20260722_112047_seed10_retry3
- `84e5cd27f679ca37c74182c58feadac43754f01bde05049886867a982a122aba` bak_20260723_105404_gpt2_rerun_lr1e-4
- `c0fb0caa776bc5ab5fe72dcde32ead66079ba0b7001cdd2657d23fe440eee6df` bak_20260724_091453_gpt2_seed5_retry
- `62b1db1c05bd80553973ee80507bd53147346fe466abcb44150be5b917e0cfeb` bak_20260724_094530_bestckpt_fix_restart
- `cc9d4ddc574314431539666f88be87c64f3e25089fa6c389306a479cde969a99` bak_20260724_164033_lost17_rerun_kick
- `36c8fac74526a1f6de3b617237276bf2976b0feea582e4d3b414689e65f5733e` bak_20260727_091301_seed8_retry42

## Not preserved in-repo (kept filesystem-only, Octopus Lustre)

Larger or externally-reproducible artifacts remain outside git management,
under `tmp/archives/` and `tmp/figures/` on the Octopus filesystem. These
are gitignored:

- `tmp/archives/molcrawl-evaluation-*.tar.gz` (~17 MB, 3 files, 2026-05 evaluation snapshots)
- `tmp/archives/OMIM.tar.gz` (~2.7 MB, 2026-05-01 OMIM dataset archive)
- `tmp/figures/*.png` (~1.8 MB, 7 generated plots; regeneration scripts committed under `scripts/`)

## Retrieval

To restore the v1 CSVs to their original path for a reproducibility check:

```bash
cd /tmp
tar xzf /path/to/repo/audit_archives/20260729_v2_csv_incident/v1_csv_snapshot.tar.gz
# extracts to ./subsets/, contents are byte-identical to the v1 state.
```

To inspect a coordinator state snapshot:

```bash
python -m json.tool \
  audit_archives/20260729_v2_csv_incident/coordinator_state_bak/coordinator_state.json.bak_20260724_094530_bestckpt_fix_restart \
  | less
```

## Not to be modified

Contents are audit artifacts. Do not amend, rebase, or edit files under this
directory. If additional artifacts need to be added for the same incident,
create a follow-up entry in this MANIFEST rather than replacing anything.
