"""Compatibility entry point for the relocated FASTA preprocessor.

New code should import ``molcrawl.preparation.genome.preprocess_fasta``.
The old command and named imports continue to work.
"""

from molcrawl.preparation.genome import preprocess_fasta as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == "__main__":
    _implementation.main()
