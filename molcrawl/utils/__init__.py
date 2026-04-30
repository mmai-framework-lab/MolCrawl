"""Backward-compatibility shim.

The canonical location is :mod:`molcrawl.core.utils`. This stub keeps the
legacy ``molcrawl.utils`` import path working during the package-layout
refactor (stage 1). It will be removed once all callers migrate.
"""

from molcrawl.core.utils import *  # noqa: F401,F403
