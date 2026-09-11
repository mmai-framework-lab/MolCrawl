"""Keep generated files out of the tree the inputs come from.

The genome data lives under another user's directory, group-readable and not
writable by us, and the same is true of any shared corpus another workstream
prepares. That arrangement has already stopped two mistakes -- a rebuild aimed at
the wrong root, and a run that would have written next to the production data --
so it is worth keeping. What it does not do is make the mistakes visible: a
write into a read-only tree fails somewhere in the middle of a job, with a
PermissionError from whatever library happened to try it.

Three places can send output into an input tree:

- ``preparation.py`` writes markers, parquet and arrow under whatever
  ``base_dir`` it is handed
- ``datasets.map`` puts its cache shards beside the dataset being read unless
  ``cache_file_name`` says otherwise -- this is the one that failed
- ``core.paths`` derives ``model_path`` from ``LEARNING_SOURCE_DIR``, so a run
  that does not override it writes checkpoints into the corpus

Check the destination up front instead, and say which input it collides with.
"""

import os
from pathlib import Path

SOURCE_ROOT_ENV = ("LEARNING_SOURCE_DIR", "GENOME_SOURCE_ROOT", "SRC_ROOT")


def _resolve(p):
    return Path(os.path.abspath(os.path.expanduser(str(p))))


def is_inside(path, root):
    """True when ``path`` is ``root`` or sits under it."""
    path, root = _resolve(path), _resolve(root)
    return path == root or root in path.parents


def input_roots(extra=()):
    """The input trees this process was pointed at, from the usual env vars."""
    roots = [os.environ[k] for k in SOURCE_ROOT_ENV if os.environ.get(k)]
    roots.extend(str(e) for e in extra if e)
    return [_resolve(r) for r in roots]


def assert_output_dir(path, extra_roots=(), what="output"):
    """Refuse an output path that lands in an input tree or cannot be written.

    Raises before any work starts, naming the input tree it collided with, so
    the failure is a sentence rather than a PermissionError from three libraries
    down.
    """
    dest = _resolve(path)
    for root in input_roots(extra_roots):
        if is_inside(dest, root):
            raise ValueError(
                f"{what} would be written inside an input tree.\n"
                f"  {what}: {dest}\n"
                f"  input : {root}\n"
                f"Point it at a directory this project owns."
            )

    probe = dest
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not os.access(probe, os.W_OK):
        raise ValueError(
            f"{what} is not writable.\n"
            f"  {what}: {dest}\n"
            f"  blocked at: {probe} (owner {probe.owner()})"
        )
    return dest


def map_cache_path(dest_dir, name="map-cache"):
    """A cache location for ``datasets.map`` that is beside the OUTPUT.

    ``map`` defaults to writing beside the input, which fails on a read-only
    corpus and pollutes it when it does not. Callers pass the result as
    ``cache_file_name``.
    """
    cache = _resolve(dest_dir).parent / f"{_resolve(dest_dir).name}.{name}"
    return cache
