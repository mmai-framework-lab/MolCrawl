"""autopilot coordinator — 4 day autonomous driver.

Runs in nohup on a login node. Polls SLURM state periodically, advances the
pipeline, writes milestone markdown, parks anything that requires human
judgment. Idempotent: state.json is the source of truth, so if the coordinator
is killed and restarted it resumes from the last recorded state.

charter §「自律判断ルール」を encode:
- realized 窓数 → aggregate_realized_windows.py (中央値近傍で target 決定)
- 発散検知 → parse_train_log_for_divergence.py (val_loss NaN or 単調悪化)
- 発散 run は park + skip、 他 modality は続行
- OpenGenome2 全データ学習は絶対に起動しない

State machine (per pipeline):
    genome_g2:
      IDLE -> STEPS1_3_QUEUED -> STEPS1_3_DONE ->
      TARGET_DECIDED -> STEP4_QUEUED -> STEP4_DONE -> DONE
    compounds_train (per config):
      IDLE -> QUEUED -> RUNNING -> DONE|PARKED
    (rna/protein/mol_nl/mammal: similar, kicked in order of priority)

Usage:
    nohup python3 -u tmp/scripts/autopilot/coordinator.py \\
        > tmp/scripts/autopilot/logs/coordinator.log 2>&1 &
    disown

Or bootstrap via ``bash tmp/scripts/autopilot/kickoff.sh``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/lustre/home/matsubara/riken-dataset-fundational-model")
AUTOPILOT = ROOT / "tmp" / "scripts" / "autopilot"
STATE_DIR = AUTOPILOT / "state"
LOGS_DIR = AUTOPILOT / "logs"
MILESTONES_DIR = AUTOPILOT / "milestones"
SBATCH_DIR = AUTOPILOT / "sbatch"
ANALYZERS_DIR = AUTOPILOT / "analyzers"

STATE_JSON = STATE_DIR / "coordinator_state.json"
PARK_LOG = MILESTONES_DIR / "park_log.md"

SUBSETS_FILE = STATE_DIR / "subsets_21.txt"

# Where compounds train.sh writes its per-run log (matches
# workflows/03[ac]-compounds-train-*.sh).
LEARNING_SOURCE_DIR = os.environ.get(
    "LEARNING_SOURCE_DIR",
    "/lustre/home/matsubara/learning_source_20260710_autopilot_v2",
)
COMPOUNDS_LOG_DIR = Path(LEARNING_SOURCE_DIR) / "compounds" / "logs"
GENOME_LOG_DIR = Path(LEARNING_SOURCE_DIR) / "genome_sequence" / "logs"

# charter §「対象は G2 → compounds/subset small → rna → mammal」
# 実運用: compounds は data 準備済ですぐ回せる → G2 と並列で kick、
# genome subset small 21 subset は G2 完了後、 mammal size axis は subset の後。
# rna, protein, mol_nl は既存 data が使える範囲で並行。

COMPOUNDS_WORKFLOWS = [
    ("bert-small", "03c-compounds-train-bert-small.sh"),
    ("gpt2-small", "03a-compounds-train-gpt2-small.sh"),
    ("bert-medium", "03c-compounds-train-bert-medium.sh"),
    ("gpt2-medium", "03a-compounds-train-gpt2-medium.sh"),
    ("bert-large", "03c-compounds-train-bert-large.sh"),
    ("gpt2-large", "03a-compounds-train-gpt2-large.sh"),
    ("gpt2-xl", "03a-compounds-train-gpt2-xl.sh"),
]

# Subset training workflows (charter § genome subset small = 論文中核).
# One workflow key per (arch, subset). Wrappers pass GENOME_SUBSET via env.
SUBSET_TRAINING_ARCHS = [
    ("bert", "03c-genome_sequence-train-bert-small-subset.sh"),
    ("gpt2", "03a-genome_sequence-train-gpt2-small-subset.sh"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_state() -> dict:
    if not STATE_JSON.exists():
        return {
            "created_at": now_iso(),
            "genome_g2": {"phase": "IDLE", "subsets": {}},
            "compounds": {},
            "bert_large_retrain": {"phase": "IDLE"},
            "subset_training": {"enabled": False, "runs": {}},
            "park": [],
        }
    return json.loads(STATE_JSON.read_text())


def save_state(state: dict) -> None:
    state["updated_at"] = now_iso()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2))


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


def sbatch_submit(sbatch: Path, job_name: str, exports: dict[str, str]) -> str | None:
    """Submit an sbatch script, return SLURM JOBID (str) or None on failure."""
    export_str = ",".join(f"{k}={v}" for k, v in exports.items())
    cmd = ["sbatch", "--parsable", f"--job-name={job_name}",
           f"--export=ALL,{export_str}", str(sbatch)]
    log(f"submit: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        log("  → sbatch timed out")
        return None
    if r.returncode != 0:
        log(f"  → sbatch failed rc={r.returncode}: {r.stderr.strip()}")
        return None
    jobid = r.stdout.strip().split(";")[0].strip()
    log(f"  → JOBID {jobid}")
    return jobid


def sacct_state(jobid: str) -> str:
    """Return SLURM job state via sacct (COMPLETED / FAILED / RUNNING / PENDING / ...)."""
    try:
        r = subprocess.run(
            ["sacct", "-j", jobid, "-n", "-o", "State", "-P", "-X"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        log(f"sacct error for {jobid}: {e}")
        return "UNKNOWN"
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        return "UNKNOWN"
    # first line is the batch step; may include suffixes like " CANCELLED+"
    return lines[0].split()[0]


def park(state: dict, item: str, reason: str) -> None:
    entry = {"at": now_iso(), "item": item, "reason": reason}
    state["park"].append(entry)
    PARK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PARK_LOG.open("a") as f:
        f.write(f"- [{entry['at']}] **{item}** — {reason}\n")
    log(f"PARK: {item} — {reason}")


def milestone(name: str, body: str) -> None:
    MILESTONES_DIR.mkdir(parents=True, exist_ok=True)
    p = MILESTONES_DIR / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    p.write_text(body)
    log(f"MILESTONE: {p.name}")


def check_run_diverged(log_glob: str) -> tuple[bool, str]:
    """Run the divergence detector on the newest log matching `log_glob`.

    Charter §「発散 park」 rule: even for SLURM COMPLETED runs, compare
    final val loss / best. Returns (diverged, message).
    """
    py = os.environ.get(
        "MOLCRAWL_PYTHON",
        "/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python",
    )
    detector = ANALYZERS_DIR / "parse_train_log_for_divergence.py"
    from glob import glob
    matches = sorted(glob(log_glob))
    if not matches:
        return (False, f"no log matching {log_glob}")
    latest = matches[-1]  # newest
    try:
        r = subprocess.run(
            [py, str(detector), latest, "--last-vs-best", "--collapse-check"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        return (False, f"detector error: {e}")
    if r.returncode == 1:
        return (True, r.stdout.strip() or "diverged")
    return (False, r.stdout.strip() or "healthy")


# ── genome G2 pipeline ─────────────────────────────────────────────────────


def kick_g2_steps1_3(state: dict) -> None:
    g2 = state["genome_g2"]
    if g2["phase"] != "IDLE":
        return
    subsets = [ln.strip() for ln in SUBSETS_FILE.read_text().splitlines()
               if ln.strip() and not ln.startswith("#")]
    for s in subsets:
        jobid = sbatch_submit(
            SBATCH_DIR / "genome_g2_subset_steps1_3.sbatch",
            job_name=f"g2-{s}",
            exports={"SUBSET": s},
        )
        if jobid is None:
            park(state, f"g2:{s}", "sbatch submit failed at IDLE → STEPS1_3")
            continue
        g2["subsets"][s] = {"phase": "STEPS1_3_QUEUED", "jobid_1_3": jobid}
    g2["phase"] = "STEPS1_3_QUEUED"
    save_state(state)


def poll_g2_steps1_3(state: dict) -> None:
    g2 = state["genome_g2"]
    if g2["phase"] != "STEPS1_3_QUEUED":
        return
    all_done = True
    for s, ss in g2["subsets"].items():
        if ss["phase"] == "STEPS1_3_QUEUED":
            st = sacct_state(ss["jobid_1_3"])
            if st == "COMPLETED":
                ss["phase"] = "STEPS1_3_DONE"
            elif st in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"):
                ss["phase"] = "STEPS1_3_FAILED"
                park(state, f"g2:{s}", f"steps 1-3 SLURM state={st}")
            elif st in ("RUNNING", "PENDING", "COMPLETING", "REQUEUED"):
                all_done = False
            else:
                all_done = False  # UNKNOWN — poll again next tick
        elif ss["phase"] == "STEPS1_3_FAILED":
            pass  # parked
        else:
            pass  # done
    if all_done:
        g2["phase"] = "STEPS1_3_DONE"
        milestone(
            "g2-steps1_3-complete",
            f"# G2 steps 1-3 完了 (autopilot)\n\n"
            f"作成: {now_iso()}\n\n"
            + "\n".join(
                f"- {s}: {ss['phase']} (jobid={ss.get('jobid_1_3')})"
                for s, ss in g2["subsets"].items()
            ),
        )
    save_state(state)


def decide_g2_target(state: dict) -> None:
    g2 = state["genome_g2"]
    if g2["phase"] != "STEPS1_3_DONE":
        return
    py = os.environ.get("MOLCRAWL_PYTHON", "/lustre/home/matsubara/miniforge3/envs/molcrawl/bin/python")
    lsd = os.environ.get(
        "LEARNING_SOURCE_DIR",
        "/lustre/home/matsubara/learning_source_20260710_genome_v2",
    )
    out_state = STATE_DIR / "g2_target.json"
    out_report = MILESTONES_DIR / "g2-target-decided.md"
    cmd = [
        py, str(ANALYZERS_DIR / "aggregate_realized_windows.py"),
        "--learning-source", lsd,
        "--subsets-file", str(SUBSETS_FILE),
        "--model", "bert",
        "--out-state", str(out_state),
        "--out-report", str(out_report),
    ]
    log(f"decide target: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        park(state, "g2:target", f"aggregate_realized_windows exit {r.returncode}: {r.stderr[:400]}")
        save_state(state)
        return
    tgt = json.loads(out_state.read_text())
    g2["target_total_windows"] = tgt["target_total_windows"]
    g2["phase"] = "TARGET_DECIDED"
    save_state(state)


def kick_g2_step4(state: dict) -> None:
    g2 = state["genome_g2"]
    if g2["phase"] != "TARGET_DECIDED":
        return
    target = str(g2["target_total_windows"])
    for s, ss in g2["subsets"].items():
        if ss["phase"] != "STEPS1_3_DONE":
            continue
        jobid = sbatch_submit(
            SBATCH_DIR / "genome_g2_subset_step4.sbatch",
            job_name=f"g2-split-{s}",
            exports={"SUBSET": s, "TARGET_TOTAL_WINDOWS": target},
        )
        if jobid is None:
            park(state, f"g2:{s}", "sbatch submit failed at TARGET_DECIDED → STEP4")
            continue
        ss["phase"] = "STEP4_QUEUED"
        ss["jobid_4"] = jobid
    g2["phase"] = "STEP4_QUEUED"
    save_state(state)


def poll_g2_step4(state: dict) -> None:
    g2 = state["genome_g2"]
    if g2["phase"] != "STEP4_QUEUED":
        return
    all_done = True
    for s, ss in g2["subsets"].items():
        if ss["phase"] == "STEP4_QUEUED":
            st = sacct_state(ss["jobid_4"])
            if st == "COMPLETED":
                ss["phase"] = "STEP4_DONE"
            elif st in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"):
                ss["phase"] = "STEP4_FAILED"
                park(state, f"g2:{s}", f"step4 SLURM state={st}")
            else:
                all_done = False
    if all_done:
        g2["phase"] = "STEP4_DONE"
        milestone(
            "g2-step4-complete",
            f"# G2 step 4 完了 (autopilot)\n\n"
            f"作成: {now_iso()}\n\n"
            f"target_total_windows: {g2.get('target_total_windows')}\n\n"
            + "\n".join(
                f"- {s}: {ss['phase']} (jobid={ss.get('jobid_4')})"
                for s, ss in g2["subsets"].items()
            ),
        )
    save_state(state)


# ── compounds pipeline ─────────────────────────────────────────────────────


def kick_compounds(state: dict, max_concurrent: int = 2) -> None:
    """Kick compounds jobs in priority order, up to max_concurrent live."""
    cmp = state["compounds"]
    for key, _wf in COMPOUNDS_WORKFLOWS:
        if key not in cmp:
            cmp[key] = {"phase": "IDLE"}
    # count live
    live = [k for k, v in cmp.items() if v.get("phase") in ("QUEUED", "RUNNING")]
    for key, wf in COMPOUNDS_WORKFLOWS:
        if len(live) >= max_concurrent:
            break
        e = cmp[key]
        if e.get("phase") == "IDLE":
            jobid = sbatch_submit(
                SBATCH_DIR / "compounds_train.sbatch",
                job_name=f"cmp-{key}",
                exports={"WORKFLOW": wf, "NUM_GPUS": "4"},
            )
            if jobid is None:
                park(state, f"compounds:{key}", "sbatch submit failed")
                e["phase"] = "PARKED"
                continue
            e["phase"] = "QUEUED"
            e["jobid"] = jobid
            e["workflow"] = wf
            live.append(key)
    save_state(state)


def poll_compounds(state: dict) -> None:
    cmp = state["compounds"]
    for key, e in cmp.items():
        if e.get("phase") in ("QUEUED", "RUNNING"):
            st = sacct_state(e["jobid"])
            if st == "COMPLETED":
                # charter §「発散 park」rule: post-COMPLETED last-vs-best check.
                # Matches log filename produced by workflows/03[ac]-compounds-train-*.sh:
                #   compounds-train-<arch>-<size>-YYYY-MM-DD_HH-MM-SS.log
                log_glob = str(COMPOUNDS_LOG_DIR / f"compounds-train-{key}-*.log")
                diverged, msg = check_run_diverged(log_glob)
                if diverged:
                    e["phase"] = "PARKED_DIVERGED"
                    park(state, f"compounds:{key}",
                         f"COMPLETED but divergence detected: {msg}")
                    milestone(
                        f"compounds-{key}-parked-diverged",
                        f"# compounds {key} 発散 park (autopilot)\n\n"
                        f"作成: {now_iso()}\n\nSLURM jobid: {e['jobid']}\n"
                        f"detector: {msg}\n",
                    )
                else:
                    e["phase"] = "DONE"
                    milestone(
                        f"compounds-{key}-done",
                        f"# compounds {key} 完了 (autopilot)\n\n"
                        f"作成: {now_iso()}\n\nSLURM jobid: {e['jobid']}\n"
                        f"workflow: {e['workflow']}\ndivergence check: {msg}\n",
                    )
            elif st in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"):
                e["phase"] = "PARKED"
                park(state, f"compounds:{key}", f"SLURM state={st} — see log")
            elif st == "RUNNING":
                e["phase"] = "RUNNING"
            # else keep current
    save_state(state)


# ── bert-large retrain (Phase 1-5b LR-ladder auto-downgrade) ───────────────


# charter 2026-07-15 reply §「早期 abort ルール」: LR ladder for bert-large.
# Start at 5e-5 (2026-07-15 replacement for the 1e-4 that plateaued),
# downgrade automatically if early plateau is detected. Match the LR values
# the config's env override accepts.
BERT_LARGE_LR_LADDER = [0.00005, 0.00003, 0.00001]  # 5e-5 → 3e-5 → 1e-5
BERT_LARGE_LR_TAGS = ["lr5e-5", "lr3e-5", "lr1e-5"]

# By eval 6, val_loss must be well below the 2.5x plateau (small/medium at
# LR 1e-4 reached 0.6-0.7 by eval 6; anything > 1.5 is the failure pattern).
EARLY_PLATEAU_MIN_EVALS = 6
EARLY_PLATEAU_THRESHOLD = 1.5


def _read_val_losses(log_path: Path) -> list[float]:
    """Same regex pair as the divergence detector — kept inline to avoid a
    subprocess call every tick."""
    import re
    text = log_path.read_text(errors="replace")
    losses: list[float] = []
    nano = re.compile(r"(?<![a-z_])val[_ ]loss\s+([\d.]+|nan)", re.IGNORECASE)
    hf = re.compile(r"['\"]eval_loss['\"]\s*:\s*([\d.]+|nan)", re.IGNORECASE)
    for line in text.splitlines():
        m = nano.search(line) or hf.search(line)
        if m:
            v = m.group(1)
            losses.append(float("nan") if v.lower() == "nan" else float(v))
    return losses


def check_early_plateau(log_glob: str) -> tuple[str, str]:
    """Return ('plateau', msg) if by eval N the val loss is still above
    threshold; ('early', msg) if not enough evals yet; ('descending', msg)
    if the run is behaving.
    """
    from glob import glob
    matches = sorted(glob(log_glob))
    if not matches:
        return ("early", "no log yet")
    log = Path(matches[-1])
    losses = _read_val_losses(log)
    if len(losses) < EARLY_PLATEAU_MIN_EVALS:
        return ("early", f"only {len(losses)} evals so far, need ≥ {EARLY_PLATEAU_MIN_EVALS}")
    at_check = losses[EARLY_PLATEAU_MIN_EVALS - 1]
    if at_check > EARLY_PLATEAU_THRESHOLD:
        return ("plateau",
                f"eval {EARLY_PLATEAU_MIN_EVALS} val_loss = {at_check:.4f} "
                f"> {EARLY_PLATEAU_THRESHOLD}; trajectory {losses[:EARLY_PLATEAU_MIN_EVALS]}")
    return ("descending", f"eval {EARLY_PLATEAU_MIN_EVALS} val_loss = {at_check:.4f} ≤ {EARLY_PLATEAU_THRESHOLD}")


def _bert_large_lr_output_dir(lr_tag: str) -> str:
    """Per-LR-attempt output dir so consecutive attempts don't clobber each other."""
    return f"/lustre/home/matsubara/learning_source_20260710_autopilot_v2_bertlarge_retrain_{lr_tag}"


def _bert_large_log_glob(lr_tag: str) -> str:
    return f"{_bert_large_lr_output_dir(lr_tag)}/compounds/logs/compounds-train-bert-large-*.log"


def kick_bert_large_retrain(state: dict) -> None:
    """Kick a fresh compounds bert-large run at the current LR-ladder step.
    Gated by state['bert_large_retrain']['enabled']. On IDLE + enabled,
    submits with the current-ladder LR (default: first ladder step = 5e-5)."""
    r = state.setdefault("bert_large_retrain", {"phase": "IDLE"})
    if not r.get("enabled"):
        return
    if r.get("phase") != "IDLE":
        return
    lr_idx = r.get("lr_idx", 0)
    if lr_idx >= len(BERT_LARGE_LR_LADDER):
        r["phase"] = "PARKED_LR_LADDER_EXHAUSTED"
        park(state, "bert_large_retrain",
             f"LR ladder {BERT_LARGE_LR_LADDER} exhausted, all plateaued")
        save_state(state)
        return
    lr = BERT_LARGE_LR_LADDER[lr_idx]
    tag = BERT_LARGE_LR_TAGS[lr_idx]
    out_dir = _bert_large_lr_output_dir(tag)
    jobid = sbatch_submit(
        SBATCH_DIR / "compounds_train.sbatch",
        job_name=f"cmp-bert-large-{tag}",
        exports={
            "WORKFLOW": "03c-compounds-train-bert-large.sh",
            "NUM_GPUS": "4",
            "LEARNING_SOURCE_DIR": out_dir,
            "SUBSET_BERT_LARGE_LR": str(lr),
        },
    )
    if jobid is None:
        park(state, "bert_large_retrain", "sbatch submit failed")
        r["phase"] = "PARKED"
    else:
        r["phase"] = "QUEUED"
        r["jobid"] = jobid
        r["current_lr"] = lr
        r["current_tag"] = tag
        r["out_dir"] = out_dir
    save_state(state)


def poll_bert_large_retrain(state: dict) -> None:
    r = state.get("bert_large_retrain", {})
    if r.get("phase") not in ("QUEUED", "RUNNING"):
        return
    st = sacct_state(r["jobid"])
    if st == "COMPLETED":
        log_glob = _bert_large_log_glob(r["current_tag"])
        diverged, msg = check_run_diverged(log_glob)
        if diverged:
            r["phase"] = "PARKED_DIVERGED"
            park(state, "bert_large_retrain",
                 f"retrain at LR={r['current_lr']} diverged: {msg}")
        else:
            r["phase"] = "DONE"
            milestone(
                "bert-large-retrain-done",
                f"# compounds bert-large retrain 完了\n\n"
                f"作成: {now_iso()}\nLR: {r['current_lr']} ({r['current_tag']})\n"
                f"SLURM jobid: {r['jobid']}\n{msg}\n",
            )
    elif st in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"):
        r["phase"] = "PARKED"
        park(state, "bert_large_retrain", f"SLURM state={st} at LR={r.get('current_lr')}")
    elif st == "RUNNING":
        r["phase"] = "RUNNING"
        # charter 2026-07-15 reply §「早期 abort」: auto-cancel + LR downgrade
        # if eval N shows the same plateau as the 1e-4 attempt.
        # Only trigger once we've observed enough evals (early-return otherwise).
        # `notified_plateau_at` tracks the previous jobid we already handled so
        # we don't fire twice for the same run.
        if r.get("notified_plateau_at") == r["jobid"]:
            return
        log_glob = _bert_large_log_glob(r["current_tag"])
        verdict, msg = check_early_plateau(log_glob)
        if verdict == "plateau":
            log(f"bert-large early plateau at LR {r['current_lr']}: {msg}")
            # Cancel current run, note the downgrade, requeue as IDLE for
            # the next tick (which advances to the next ladder step).
            subprocess.run(["scancel", r["jobid"]], capture_output=True, timeout=30)
            history = r.setdefault("ladder_history", [])
            history.append({
                "jobid": r["jobid"], "lr": r["current_lr"],
                "reason": "early plateau", "detail": msg, "at": now_iso(),
            })
            park(state, "bert_large_retrain",
                 f"auto-kill @ LR={r['current_lr']}: {msg}")
            r["notified_plateau_at"] = r["jobid"]
            r["lr_idx"] = r.get("lr_idx", 0) + 1
            r["phase"] = "IDLE"
    save_state(state)


# ── subset training (21 subset × 2 arch = 42 run) ──────────────────────────


def _subset_keys() -> list[str]:
    return [ln.strip() for ln in SUBSETS_FILE.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def kick_subset_training(state: dict, max_concurrent: int = 10) -> None:
    """Kick 21 subset × 2 arch = 42 runs, guarded by explicit enable flag.

    The user (or a `readiness_go.py` script) sets
    ``state['subset_training']['enabled'] = True`` after the readiness
    report is approved. Until then this is a no-op.

    Boss reply 2026-07-16: 4 → 10 concurrent so we saturate the newly
    provisioned H200 fleet (24 current + 16 pending = 40 → 10 × 4 = 40 GPU).
    Excess submissions queue at SLURM level as Resources when the extra
    16 GPU come online.
    """
    sub = state.setdefault("subset_training", {"enabled": False, "runs": {}})
    if not sub.get("enabled"):
        return

    runs = sub["runs"]
    # Prime IDLE entries for every (arch, subset) pair we haven't seen.
    subsets = _subset_keys()
    for arch, wf in SUBSET_TRAINING_ARCHS:
        for s in subsets:
            key = f"{arch}-{s}"
            if key not in runs:
                runs[key] = {"phase": "IDLE", "arch": arch, "subset": s, "workflow": wf}

    live = [k for k, v in runs.items() if v.get("phase") in ("QUEUED", "RUNNING")]
    for key, e in runs.items():
        if len(live) >= max_concurrent:
            break
        if e.get("phase") != "IDLE":
            continue
        jobid = sbatch_submit(
            SBATCH_DIR / "subset_train.sbatch",
            job_name=f"sub-{key}",
            exports={
                "WORKFLOW": e["workflow"],
                "GENOME_SUBSET": e["subset"],
                "NUM_GPUS": "4",
                # subset training reads from the G2 output tree.
                "LEARNING_SOURCE_DIR":
                    "/lustre/home/matsubara/learning_source_20260710_genome_v2",
            },
        )
        if jobid is None:
            park(state, f"subset:{key}", "sbatch submit failed")
            e["phase"] = "PARKED"
            continue
        e["phase"] = "QUEUED"
        e["jobid"] = jobid
        live.append(key)
    save_state(state)


def poll_subset_training(state: dict) -> None:
    sub = state.get("subset_training", {"enabled": False, "runs": {}})
    runs = sub.get("runs", {})
    for key, e in runs.items():
        if e.get("phase") not in ("QUEUED", "RUNNING"):
            continue
        st = sacct_state(e["jobid"])
        if st == "COMPLETED":
            arch = e["arch"]
            subset = e["subset"]
            # workflows/03[ac]-genome_sequence-train-<arch>-small-subset.sh writes to
            # $LEARNING_SOURCE_DIR/genome_sequence/logs/<subset>-<arch>-small-*.log
            log_glob = str(
                Path("/lustre/home/matsubara/learning_source_20260710_genome_v2")
                / "genome_sequence" / "logs" / f"{subset}-{arch}-small-*.log"
            )
            diverged, msg = check_run_diverged(log_glob)
            if diverged:
                e["phase"] = "PARKED_DIVERGED"
                park(state, f"subset:{key}",
                     f"COMPLETED but divergence: {msg}")
            else:
                e["phase"] = "DONE"
                milestone(
                    f"subset-{key}-done",
                    f"# subset {key} 完了 (autopilot)\n\n"
                    f"作成: {now_iso()}\nSLURM jobid: {e['jobid']}\n"
                    f"divergence check: {msg}\n",
                )
        elif st in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"):
            e["phase"] = "PARKED"
            park(state, f"subset:{key}", f"SLURM state={st}")
        elif st == "RUNNING":
            e["phase"] = "RUNNING"
    save_state(state)


# ── main loop ──────────────────────────────────────────────────────────────


def one_tick(state: dict) -> None:
    # G2 pipeline advancement
    kick_g2_steps1_3(state)
    poll_g2_steps1_3(state)
    decide_g2_target(state)
    kick_g2_step4(state)
    poll_g2_step4(state)
    # Compounds (parallel, priority-ordered)
    kick_compounds(state)
    poll_compounds(state)
    # bert-large retrain (gated by state['bert_large_retrain']['enabled'])
    kick_bert_large_retrain(state)
    poll_bert_large_retrain(state)
    # Subset training (gated by state['subset_training']['enabled'])
    kick_subset_training(state)
    poll_subset_training(state)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick-seconds", type=int, default=300,
                    help="poll interval (default 300s)")
    ap.add_argument("--max-ticks", type=int, default=None,
                    help="stop after N ticks (default: run forever)")
    ap.add_argument("--once", action="store_true",
                    help="run one tick and exit (for cron/testing)")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MILESTONES_DIR.mkdir(parents=True, exist_ok=True)

    log(f"coordinator start, pid={os.getpid()}, tick={args.tick_seconds}s")

    ticks = 0
    while True:
        try:
            state = load_state()
            one_tick(state)
        except Exception as e:
            import traceback
            log(f"tick error: {e}\n{traceback.format_exc()}")
        ticks += 1
        if args.once or (args.max_ticks and ticks >= args.max_ticks):
            log(f"coordinator exit after {ticks} tick(s)")
            return 0
        time.sleep(args.tick_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
