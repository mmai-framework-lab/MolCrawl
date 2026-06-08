"""R3: stand-alone shared tokenizer trainer with RAM monitoring.

Runs the same train_shared_tokenizer logic the prepare-local pipeline uses,
but as an isolated process so RAM can be watched in real time. Used to
recover from the 2026-05-19 batch failure where SentencePiece died silently
during Stage 3 with input_sentence_size=700000.

Reduced sample (matching R2's yaml edit): input_sentence_size=200000.

Exits early (no-op) if /lustre/home/kojima-t/data/species_links/spm_tokenizer.model
already exists — so this can race with the prepare-local restart (R2) without
clobbering whichever finishes first.

Writes per-15s RAM snapshots to <ram_log> for live diagnosis.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# Ensure LEARNING_SOURCE_DIR is set for molcrawl imports
os.environ.setdefault("LEARNING_SOURCE_DIR", "../learning_source_20260518_genome")


def ram_monitor(log_path: Path, stop_event: threading.Event) -> None:
    import psutil
    proc = psutil.Process()
    with open(log_path, "a") as f:
        f.write(f"# R3 RAM monitor started at {time.strftime('%Y-%m-%d %H:%M:%S')}, PID={os.getpid()}\n")
        f.flush()
        while not stop_event.is_set():
            vm = psutil.virtual_memory()
            try:
                rss = proc.memory_info().rss / 1e9
            except psutil.NoSuchProcess:
                break
            f.write(
                f"{time.strftime('%H:%M:%S')}  proc_rss={rss:6.2f}G  sys_used={vm.used/1e9:6.2f}G  sys_avail={vm.available/1e9:6.2f}G\n"
            )
            f.flush()
            stop_event.wait(15)


def main() -> None:
    species_links = Path(
        os.environ.get("SPECIES_LINKS_DIR", "/lustre/home/kojima-t/data/species_links")
    )
    model_path = species_links / "spm_tokenizer.model"
    if model_path.exists():
        print(f"R3: tokenizer already exists at {model_path}, exiting (R2 likely won the race).")
        return

    ram_log = Path(os.environ["LEARNING_SOURCE_DIR"]) / "genome_sequence" / "logs" / "r3_ram_monitor.log"
    ram_log.parent.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    monitor_thread = threading.Thread(target=ram_monitor, args=(ram_log, stop), daemon=True)
    monitor_thread.start()

    try:
        from molcrawl.data.genome_sequence import preparation_local as pl
        from molcrawl.data.genome_sequence.utils.config import GenomeSequenceConfig

        cfg = GenomeSequenceConfig.from_file("assets/configs/genome_sequence.yaml").data_preparation
        species_dirs = [out for _, out in pl.find_species_inputs(species_links)]
        print(f"R3: training tokenizer across {len(species_dirs)} species...")
        print(f"R3: vocab_size={cfg.vocab_size}, input_sentence_size=200000 (override), max_lines_per_file={cfg.max_lines_per_file}")
        print(f"R3: RAM monitor → {ram_log}")

        pl.train_shared_tokenizer(
            species_dirs,
            species_links,
            vocab_size=cfg.vocab_size,
            max_lines_per_file=cfg.max_lines_per_file,
            input_sentence_size=200000,
            force=False,
        )
        print(f"R3: tokenizer written → {model_path}")
    finally:
        stop.set()
        monitor_thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
