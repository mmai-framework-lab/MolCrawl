"""Foldability evaluation does not require a dataset split."""

from __future__ import annotations

from typing import Iterable


def dedupe_generated(sequences: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for seq in sequences:
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out
