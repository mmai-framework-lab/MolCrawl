"""Cross-task snapshot aggregator.

Reads the per-task ``metrics.json`` artefacts emitted by
:class:`molcrawl.tasks.evaluation._base.report_writer.ReportWriter`, then
produces the 3-axis (modality x arch x task) rollup required by Phase 6
of the evaluator implementation plan.

Public API:

* :func:`collect_results` - walk an evaluation output root and return a
  list of per-run dicts.
* :func:`build_snapshot` - turn the collected runs into the snapshot
  payload (JSON) plus a markdown report.
* :func:`diff_with_previous` - compute metric deltas against an earlier
  snapshot when one is provided.
"""

from .aggregator import (
    build_snapshot,
    collect_results,
    diff_with_previous,
    write_snapshot,
)

__all__ = [
    "build_snapshot",
    "collect_results",
    "diff_with_previous",
    "write_snapshot",
]
