"""Save a tokenizer so that other processes never read a half-written file.

Several configs build a tokenizer at import, save it to a fixed directory and read
it straight back. ``save_pretrained`` opens each file for writing in place, so a
second run starting at the same moment can open ``tokenizer.json`` after it has
been truncated and before it has been written, and fails on an empty JSON. That is
what killed two of three rna arms launched together (117320, 117321), and a
48-execution stress test reproduces it six times over.

Writing the directory once in advance does not help: every run writes it again.
What helps is never exposing a partial file. Each file is written in a private
directory on the same filesystem and moved into place with ``os.replace``, which
swaps the name atomically, so a reader sees either the previous complete file or
the new complete one. Concurrent runs write identical content, so which one wins
does not matter.

A directory that has never been populated can still be read while only some of
its files have arrived. Write it once before launching runs together
(``workflows/bert-prepare-tokenizer.sbatch``); after that every file is always
present and always complete.
"""

from __future__ import annotations

import os
import shutil
import tempfile


def save_pretrained_atomic(tokenizer, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    parent = os.path.dirname(os.path.abspath(path)) or "."
    tmp = tempfile.mkdtemp(prefix=f".{os.path.basename(path)}.tmp-", dir=parent)
    try:
        tokenizer.save_pretrained(tmp)
        for name in sorted(os.listdir(tmp)):
            os.replace(os.path.join(tmp, name), os.path.join(path, name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
