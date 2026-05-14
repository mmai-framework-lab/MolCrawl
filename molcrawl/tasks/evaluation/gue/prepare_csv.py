"""Materialise GUE sub-task CSVs from a HuggingFace dataset repo.

Two upstream repos host the GUE benchmark:

* ``leannmlindsey/GUE`` — community mirror, public, no token required.
  Naming: ``GUE/emp_H3K4me1/train.csv``, ``GUE/human_tf_0/train.csv``,
  ``GUE/virus_covid/train.csv``, etc.
* ``zhihan1996/DNABERT_2`` — canonical release, gated. Naming: same
  basenames as the mirror's ``GUE/<task>/`` subtree once the released
  ``GUE.zip`` is unpacked.

This module:

1. Calls :func:`huggingface_hub.snapshot_download` to clone the repo
   into a local cache (HF_TOKEN is honoured automatically when the
   repo is gated; ungated repos require no auth).
2. Locates the GUE/ root inside the cache (the mirror nests a ``GUE/``
   directory; some mirrors don't).
3. Aliases mirror sub-task names to the canonical 28-task names
   expected by :mod:`molcrawl.tasks.evaluation.gue.data_preparation`.
4. Copies the renamed sub-task directories into ``output_dir``,
   skipping any unchanged files (idempotent).

The rest of the pipeline does not need to know which upstream repo
was used — it just sees the canonical layout under
``$LEARNING_SOURCE_DIR/eval/gue/<task>/{train,dev,test}.csv``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Mirror task name → canonical name expected by data_preparation.TASKS.
# The mirror serves a superset of 37 task dirs; only the 28 in this map
# are materialised by default (extras are ignored unless --include-extras).
_MIRROR_TO_CANONICAL: Dict[str, str] = {
    "prom_300_all": "prom_300_all",
    "prom_300_notata": "prom_300_notata",
    "prom_300_tata": "prom_300_tata",
    "prom_core_all": "prom_core_all",
    "prom_core_notata": "prom_core_notata",
    "prom_core_tata": "prom_core_tata",
    "splice_reconstructed": "splice_reconstructed",
    "virus_covid": "covid_variants",
    "mouse_0": "mouse_0",
    "mouse_1": "mouse_1",
    "mouse_2": "mouse_2",
    "mouse_3": "mouse_3",
    "mouse_4": "mouse_4",
    "emp_H3": "H3",
    "emp_H3K14ac": "H3K14ac",
    "emp_H3K36me3": "H3K36me3",
    "emp_H3K4me1": "H3K4me1",
    "emp_H3K4me2": "H3K4me2",
    "emp_H3K4me3": "H3K4me3",
    "emp_H3K79me3": "H3K79me3",
    "emp_H3K9ac": "H3K9ac",
    "emp_H4": "H4",
    "emp_H4ac": "H4ac",
    "human_tf_0": "tf_0",
    "human_tf_1": "tf_1",
    "human_tf_2": "tf_2",
    "human_tf_3": "tf_3",
    "human_tf_4": "tf_4",
}


def _copy_tree(src: Path, dst: Path) -> int:
    """Copy ``src`` directory contents into ``dst``, overwriting files.

    Returns the number of files copied (excluding hidden files).
    """
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in src.rglob("*"):
        if path.name.startswith("."):
            continue
        if path.is_file():
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            n += 1
    return n


def materialise_gue(
    hf_repo: str,
    output_dir: Path,
    hf_revision: str = "main",
    include_extras: bool = False,
) -> dict:
    """Snapshot-download ``hf_repo`` and copy renamed sub-tasks into ``output_dir``."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required; install via `pip install huggingface_hub`."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "snapshot_download(repo=%s, revision=%s) ...",
        hf_repo,
        hf_revision,
    )
    cache_path = snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        revision=hf_revision,
        allow_patterns=["GUE/*/*.csv", "*.csv"],
    )
    cache = Path(cache_path)
    logger.info("snapshot at %s", cache)

    # Locate GUE/<task>/ root
    if (cache / "GUE").is_dir():
        gue_root = cache / "GUE"
    else:
        gue_root = cache

    available_dirs = sorted(p for p in gue_root.iterdir() if p.is_dir())
    logger.info(
        "found %d sub-task directories under %s",
        len(available_dirs),
        gue_root,
    )

    copied: List[str] = []
    skipped: List[str] = []
    extras: List[str] = []
    for d in available_dirs:
        mirror_name = d.name
        if mirror_name in _MIRROR_TO_CANONICAL:
            canonical = _MIRROR_TO_CANONICAL[mirror_name]
        elif mirror_name in _MIRROR_TO_CANONICAL.values():
            # Already canonical (e.g. zhihan1996/DNABERT_2 release uses canonical names directly)
            canonical = mirror_name
        else:
            extras.append(mirror_name)
            if not include_extras:
                continue
            canonical = mirror_name

        target = output_dir / canonical
        # Verify train.csv presence before copying so we can log meaningfully.
        if not (d / "train.csv").exists():
            skipped.append(mirror_name)
            continue
        n_files = _copy_tree(d, target)
        copied.append(f"{mirror_name}->{canonical} ({n_files} files)")

    summary = {
        "hf_repo": hf_repo,
        "hf_revision": hf_revision,
        "output_dir": str(output_dir),
        "n_copied": len(copied),
        "copied": copied,
        "skipped_no_train": skipped,
        "ignored_extras": extras,
    }
    logger.info(
        "materialise_gue done: copied=%d skipped=%d extras_ignored=%d",
        len(copied),
        len(skipped),
        len(extras),
    )
    if skipped:
        logger.warning("skipped (no train.csv): %s", skipped)
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Materialise GUE sub-task CSVs from a HuggingFace dataset repo"
    )
    parser.add_argument(
        "--hf-repo",
        default="leannmlindsey/GUE",
        help="Dataset repo id (default: leannmlindsey/GUE — public mirror, no token needed).",
    )
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help="Also copy mirror-only extra tasks (EPI_*, fungi_species_20, "
        "phage_fragments, virus_species_40) into output_dir.",
    )
    args = parser.parse_args(argv)

    materialise_gue(
        hf_repo=args.hf_repo,
        output_dir=Path(args.output_dir),
        hf_revision=args.hf_revision,
        include_extras=args.include_extras,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
