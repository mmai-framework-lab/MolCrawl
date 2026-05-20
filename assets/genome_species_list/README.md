# Genome species lists

This directory holds the species lists consumed by the RefSeq downloader at
[molcrawl/data/genome_sequence/dataset/refseq/download_refseq.py](../../molcrawl/data/genome_sequence/dataset/refseq/download_refseq.py).

[assets/configs/genome_sequence.yaml](../configs/genome_sequence.yaml) points
`data_preparation.path_species` at one of the two subdirectories below.

## `species/` — full intended list

Reference / aspirational catalogue. One file per `ncbi-genome-download` group
(`bacteria`, `fungi`, `invertebrate`, `protozoa`, `vertebrate_mammalian`,
`vertebrate_other`, …). One species per line, written as the canonical Latin
binomial. Used as the source of truth for what we *want* in the pretrain
corpus.

## `filtered_species_refseq/` — active list, post-failure filtering

This is the list the pipeline actually downloads. It is a *subset* of
`species/` with entries that consistently fail in `ncbi_genome_download`
removed, plus species we have decided not to attempt for the current run.

### Operating policy

1. **Add to `species/` first.** When introducing a new species, edit
   `species/<group>.txt` so the catalogue stays complete.
2. **Mirror to `filtered_species_refseq/` once verified.** Add the species
   here after a successful download (the per-species marker under
   `${LEARNING_SOURCE_DIR}/genome_sequence/download_dir/<group>/<species>/`
   confirms it).
3. **Removing entries from `filtered_species_refseq/` is allowed only after
   recording the reason** in `${LEARNING_SOURCE_DIR}/genome_sequence/failed_species.json`
   (auto-written) or in a doc under `docs/_tmp/`. Do not remove a species from
   `species/` — that file should always reflect the full intended scope.
4. **Vertebrate / mammalian coverage is load-bearing.** Removing
   `Homo sapiens`, `Mus musculus`, `Rattus norvegicus`, or any of the great
   apes / common model mammals requires an explicit corpus-design decision,
   not just a "this species failed" reaction.

### Current (2026-05-13) composition

`filtered_species_refseq/` now restores human and other key model mammals
that were dropped during earlier failure-driven pruning:

- `vertebrate_mammalian.txt`: Homo sapiens, Mus musculus, Rattus norvegicus,
  Macaca mulatta, Pan troglodytes, Canis lupus familiaris, Sus scrofa,
  Bos taurus, Bubalus bubalis, Camelus dromedarius, Peromyscus californicus.
- `vertebrate_other.txt`: Danio rerio, Gallus gallus, Xenopus tropicalis,
  Coregonus clupeaformis, Myxocyprinus asiaticus.
- `invertebrate.txt`: Drosophila melanogaster, Caenorhabditis elegans.
- `fungi.txt` / `protozoa.txt`: unchanged from the previous run.
- `bacteria.txt`: **intentionally absent from `filtered_species_refseq/`.**
  The previous run downloaded 133 GB of bacterial assemblies (39 genera ×
  many strains each) which dominated the corpus over mammalian sequence.
  Until the bacteria pipeline is switched from genus-level to
  species/representative-level queries (B-1/B-2 in
  [docs/_tmp/20260513-genome-protein-molnatlang-corpus-sizes.md](../../docs/_tmp/20260513-genome-protein-molnatlang-corpus-sizes.md)),
  bacteria are excluded from active downloads. The canonical catalogue
  `species/bacteria.txt` is retained for the future re-enable.

Background and rationale: [docs/_tmp/20260513-genome-protein-molnatlang-corpus-sizes.md](../../docs/_tmp/20260513-genome-protein-molnatlang-corpus-sizes.md).
