"""OMIM reuses the chromosome-aware split from ClinVar."""

from molcrawl.tasks.evaluation.clinvar.splits import chromosome_split

__all__ = ["chromosome_split"]
