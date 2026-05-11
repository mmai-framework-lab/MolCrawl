"""Dual JSON + markdown report writer.

Every evaluator produces two artefacts under ``output_dir``:

* ``metrics.json`` - machine readable snapshot that downstream scripts
  (weekly roll-up, dashboards) can re-read.
* ``REPORT.md`` - human readable summary with the 3-axis header
  (modality / arch / task) and the metrics table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class ReportWriter:
    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        task: str,
        modality: str,
        arch: str,
        category: str,
        metrics: Dict[str, float],
        report: Dict[str, Any],
    ) -> Dict[str, str]:
        """Write ``metrics.json`` and ``REPORT.md``.

        Returns a mapping with the paths written, which tasks surface to
        callers as ``EvaluationResult.report_paths``.
        """
        generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        payload = {
            "generated_at": generated_at,
            "task": task,
            "modality": modality,
            "arch": arch,
            "category": category,
            "metrics": {k: _coerce_scalar(v) for k, v in metrics.items()},
            "details": report,
        }

        json_path = self.output_dir / "metrics.json"
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        md_path = self.output_dir / "REPORT.md"
        md_path.write_text(_render_markdown(payload), encoding="utf-8")

        return {"json": str(json_path), "markdown": str(md_path)}


def _coerce_scalar(value: Any) -> Any:
    """Coerce common numeric wrappers (numpy scalars etc.) to plain floats."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None

    if np is not None:
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _render_markdown(payload: Dict[str, Any]) -> str:
    header = (
        f"# Evaluation report\n\n"
        f"- task: `{payload['task']}`\n"
        f"- modality: `{payload['modality']}`\n"
        f"- arch: `{payload['arch']}`\n"
        f"- category: `{payload['category']}`\n"
        f"- generated_at: `{payload['generated_at']}`\n\n"
        "## Metrics\n\n"
    )
    rows = ["| metric | value |", "| --- | --- |"]
    for name, value in payload["metrics"].items():
        rows.append(f"| {name} | {_format_value(value)} |")
    metrics_table = "\n".join(rows) + "\n"

    details = payload.get("details") or {}
    notes = ""
    if details:
        notes = "\n## Details\n\n```json\n" + json.dumps(details, indent=2, ensure_ascii=False) + "\n```\n"
    return header + metrics_table + notes


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
