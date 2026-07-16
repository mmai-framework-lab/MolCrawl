#!/bin/bash
# autopilot kickoff — launches the coordinator in nohup + writes the initial
# state file. Idempotent: safe to re-run (won't double-submit; coordinator
# reads state.json to know what's already running).

set -euo pipefail

AUTOPILOT=/lustre/home/matsubara/riken-dataset-fundational-model/tmp/scripts/autopilot
LOG="${AUTOPILOT}/logs/coordinator.log"
PID_FILE="${AUTOPILOT}/state/coordinator.pid"

mkdir -p "${AUTOPILOT}/state" "${AUTOPILOT}/logs" "${AUTOPILOT}/milestones"

# Already running? Bail out.
if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "coordinator already running (pid ${PID}). To restart, kill it first:"
        echo "  kill ${PID} && rm ${PID_FILE}"
        exit 0
    else
        echo "stale pid file (${PID} not alive) — removing"
        rm "${PID_FILE}"
    fi
fi

PY="${MOLCRAWL_PYTHON:-/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python}"
echo "starting coordinator with ${PY}"

# Use setsid + nohup so the process outlives the login shell.
nohup setsid "${PY}" -u "${AUTOPILOT}/coordinator.py" \
    --tick-seconds 300 \
    > "${LOG}" 2>&1 &
PID=$!
disown || true
echo "${PID}" > "${PID_FILE}"

echo "coordinator started: pid ${PID}"
echo "log: ${LOG}"
echo "state: ${AUTOPILOT}/state/coordinator_state.json"
echo "milestones: ${AUTOPILOT}/milestones/"
echo ""
echo "To watch: tail -f ${LOG}"
echo "To stop:  kill \$(cat ${PID_FILE}) && rm ${PID_FILE}"
