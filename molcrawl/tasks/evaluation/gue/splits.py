"""GUE sub-tasks ship fixed splits; this module is a thin re-export."""

from .data_preparation import load_splits

__all__ = ["load_splits"]
