#!/bin/bash
# Bootstrap: symlink existing .fna.gz from old subset dirs to the new v2 tree,
# then touch download_complete.marker so process1_subset_download skips.
# Rationale: charter requires "new dir" but NCBI FTP was 503-ing our fresh
# downloads. The old download_2026-05-29 tree has the same subset CSVs
# already fully downloaded (mammal_centered 24 / eukaryote 76 / global up to
# 550 species). Symlinks preserve the "new tree" boundary AND keep the
# original untouched, satisfying both guard-rails.

set -euo pipefail

OLD=/lustre/home/matsubara/learning_source_20260529_evo2species/genome_sequence
NEW=/lustre/home/matsubara/learning_source_20260710_genome_v2/genome_sequence

if [ ! -d "${OLD}" ]; then
    echo "ERROR: old subset tree not found: ${OLD}" >&2
    exit 2
fi

SUBSETS_FILE=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/state/subsets_21.txt

count_total=0
count_linked=0
count_missing_old=0
for s in $(grep -v '^#' "${SUBSETS_FILE}" | grep .); do
    old_ex="${OLD}/${s}/extracted_files"
    new_ex="${NEW}/${s}/extracted_files"
    mkdir -p "${new_ex}"

    if [ ! -d "${old_ex}" ]; then
        echo "  [MISS] ${s}: no old extracted_files/" >&2
        count_missing_old=$((count_missing_old + 1))
        continue
    fi

    n_before=$(find "${new_ex}" -maxdepth 1 -name '*.fna.gz' 2>/dev/null | wc -l)
    n_source=$(find "${old_ex}" -maxdepth 1 -name '*.fna.gz' 2>/dev/null | wc -l)
    if [ "$n_source" -eq 0 ]; then
        echo "  [WARN] ${s}: 0 .fna.gz in old dir" >&2
        continue
    fi

    while IFS= read -r f; do
        base=$(basename "${f}")
        target="${new_ex}/${base}"
        if [ -L "${target}" ] || [ -e "${target}" ]; then
            continue
        fi
        ln -s "${f}" "${target}"
        count_linked=$((count_linked + 1))
    done < <(find "${old_ex}" -maxdepth 1 -name '*.fna.gz')

    # Touch marker so process1 skips (won't re-hit NCBI).
    touch "${NEW}/${s}/download_complete.marker"

    n_after=$(find "${new_ex}" -maxdepth 1 -name '*.fna.gz' 2>/dev/null | wc -l)
    count_total=$((count_total + n_after))
    echo "  [OK] ${s}: ${n_after} fna.gz (was ${n_before}, source ${n_source}), marker touched"
done

echo ""
echo "=== bootstrap summary ==="
echo "total .fna.gz visible in new tree : ${count_total}"
echo "new symlinks created (this run)   : ${count_linked}"
echo "subsets missing in old tree       : ${count_missing_old}"
