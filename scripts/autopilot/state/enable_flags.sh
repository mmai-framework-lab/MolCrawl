#!/bin/bash
# Enable autopilot launch gates after readiness GO.
#
# Usage:
#   bash tmp/scripts/autopilot/state/enable_flags.sh bert_large
#   bash tmp/scripts/autopilot/state/enable_flags.sh subset
#   bash tmp/scripts/autopilot/state/enable_flags.sh both
#
# Idempotent: setting an already-true flag is a no-op.
# Next coord tick (≤ 5 min) picks up the change and starts submitting jobs.

set -euo pipefail

STATE=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/state/coordinator_state.json
PY=/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python

TARGET="${1:-}"
case "${TARGET}" in
    bert_large|subset|both) ;;
    *)
        echo "Usage: $0 {bert_large|subset|both}" >&2
        exit 1
        ;;
esac

${PY} - <<PYEOF
import json
from pathlib import Path
p = Path("${STATE}")
d = json.loads(p.read_text())

target = "${TARGET}"
changes = []

if target in ("bert_large", "both"):
    d.setdefault("bert_large_retrain", {"phase": "IDLE"})
    was = d["bert_large_retrain"].get("enabled", False)
    d["bert_large_retrain"]["enabled"] = True
    changes.append(f"  bert_large_retrain.enabled: {was} → True")

if target in ("subset", "both"):
    d.setdefault("subset_training", {"enabled": False, "runs": {}})
    was = d["subset_training"].get("enabled", False)
    d["subset_training"]["enabled"] = True
    changes.append(f"  subset_training.enabled: {was} → True")

from datetime import datetime, timezone
d.setdefault("notes", []).append({
    "at": datetime.now(timezone.utc).astimezone().isoformat(),
    "note": f"enable_flags.sh {target}: " + "; ".join(c.strip() for c in changes),
})

p.write_text(json.dumps(d, indent=2))
print("Flags updated:")
for c in changes:
    print(c)
print()
print("Next coord tick (≤ 5 min) will submit jobs. Watch:")
print("  tail -f /lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot/logs/coordinator.log")
print("  squeue -u matsubara")
PYEOF
