"""Snapshot aggregator for the 3-axis evaluation rollup."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunEntry:
    """Parsed per-run record."""

    task: str
    modality: str
    arch: str
    category: str
    metrics: Dict[str, float]
    source_path: str
    generated_at: Optional[str] = None

    def key(self) -> tuple:
        return (self.modality, self.arch, self.task)


def collect_results(
    root: Path,
    metrics_filename: str = "metrics.json",
) -> List[RunEntry]:
    """Walk ``root`` and load every ``metrics.json`` encountered."""
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(root)
    entries: List[RunEntry] = []
    for path in base.rglob(metrics_filename):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.warning("Skipping %s (invalid JSON: %s)", path, exc)
            continue
        try:
            entry = RunEntry(
                task=str(payload["task"]),
                modality=str(payload["modality"]),
                arch=str(payload["arch"]),
                category=str(payload.get("category", "other")),
                metrics=dict(payload.get("metrics", {})),
                source_path=str(path),
                generated_at=payload.get("generated_at"),
            )
        except KeyError as exc:  # pragma: no cover - defensive
            logger.warning("Skipping %s (missing key: %s)", path, exc)
            continue
        entries.append(entry)
    return entries


def build_snapshot(entries: Iterable[RunEntry]) -> Dict[str, Any]:
    """Turn the run list into a snapshot payload.

    Keyed by ``(modality, arch, task)`` so the weekly workflow can
    render a stable 3-axis table.  Later runs overwrite earlier runs
    for the same key (the newer one is kept).
    """
    keyed: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        existing = keyed.get("/".join(entry.key()))
        if existing and existing.get("generated_at") and entry.generated_at:
            if existing["generated_at"] >= entry.generated_at:
                continue
        keyed["/".join(entry.key())] = {
            "task": entry.task,
            "modality": entry.modality,
            "arch": entry.arch,
            "category": entry.category,
            "metrics": entry.metrics,
            "source": entry.source_path,
            "generated_at": entry.generated_at,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": sorted(keyed.values(), key=lambda r: (r["modality"], r["arch"], r["task"])),
    }


def diff_with_previous(
    current: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Return ``{run_key: {metric: delta}}`` pairs."""
    if not previous:
        return {}
    current_by_key = {"/".join((r["modality"], r["arch"], r["task"])): r for r in current.get("runs", [])}
    previous_by_key = {"/".join((r["modality"], r["arch"], r["task"])): r for r in previous.get("runs", [])}
    diff: Dict[str, Dict[str, float]] = {}
    for key, current_run in current_by_key.items():
        previous_run = previous_by_key.get(key)
        if not previous_run:
            continue
        entry_diff: Dict[str, float] = {}
        for metric, value in current_run.get("metrics", {}).items():
            prev_value = previous_run.get("metrics", {}).get(metric)
            if prev_value is None:
                continue
            try:
                entry_diff[metric] = float(value) - float(prev_value)
            except (TypeError, ValueError):
                continue
        if entry_diff:
            diff[key] = entry_diff
    return diff


def write_snapshot(
    snapshot: Mapping[str, Any],
    output_dir: Path,
    previous_snapshot: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
) -> Dict[str, str]:
    """Persist JSON + markdown snapshot under ``output_dir``.

    Returns the written paths for programmatic callers.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = name or datetime.now(timezone.utc).strftime("%Y%m%d")
    json_path = out_dir / f"snapshot_{stamp}.json"
    md_path = out_dir / f"snapshot_{stamp}.md"

    diff = diff_with_previous(snapshot, previous_snapshot)
    payload = dict(snapshot)
    payload["diff_against_previous"] = diff

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path)}


def load_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_markdown(payload: Mapping[str, Any]) -> str:
    runs = payload.get("runs", [])
    diff = payload.get("diff_against_previous", {}) or {}

    lines = [
        "# Evaluation snapshot",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- total runs: {len(runs)}",
        "",
        "## 3-axis summary (modality x arch x task)",
        "",
        "| modality | arch | task | metric | value | delta |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        key = "/".join((run["modality"], run["arch"], run["task"]))
        run_diff = diff.get(key, {})
        metrics = run.get("metrics") or {}
        if not metrics:
            lines.append(
                f"| {run['modality']} | {run['arch']} | {run['task']} | - | - | - |"
            )
            continue
        for metric, value in metrics.items():
            delta = run_diff.get(metric)
            delta_str = "-" if delta is None else f"{delta:+.4f}"
            lines.append(
                f"| {run['modality']} | {run['arch']} | {run['task']} | {metric} | {_fmt(value)} | {delta_str} |"
            )

    if diff:
        lines.extend(["", "## Movement vs previous snapshot", ""])
        movers = sorted(
            ((key, metric, delta) for key, metrics in diff.items() for metric, delta in metrics.items()),
            key=lambda item: abs(item[2]),
            reverse=True,
        )[:20]
        if movers:
            lines.append("| run | metric | delta |")
            lines.append("| --- | --- | --- |")
            for key, metric, delta in movers:
                lines.append(f"| {key} | {metric} | {delta:+.4f} |")

    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
