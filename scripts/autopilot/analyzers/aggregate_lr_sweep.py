"""Aggregate mini LR sweep results and pick the largest stable LR per arch.

charter 2026-07-14 reply §「合格条件: loss が ln(4)≒1.386 を半歩下回って滑らかに
下がる。上への spike が無い。 collapse (ln(4) 近辺で平ら)・spike は不合格。合格の
中で最大の安定 LR を採用 (arch ごと)」

Reads the sweep jobid roster (state/lr_sweep_jobids.txt), locates each run's
training log under $LEARNING_SOURCE_DIR/genome_sequence/logs/, classifies:
  - PASS   — loss went below ln(4)/2 + no spike + no plateau
  - COLLAPSE — flat near ln(4)
  - SPIKE    — max after warmup > min * spike_ratio
  - NAN      — NaN detected
  - INCOMPLETE — no eval log lines (job didn't reach an eval)

Picks the largest PASSing LR per (arch, subset) and emits a markdown report.

Usage:
    python3 tmp/scripts/autopilot/analyzers/aggregate_lr_sweep.py \
        --roster tmp/scripts/autopilot/state/lr_sweep_jobids.txt \
        --learning-source /lustre/home/matsubara/learning_source_20260710_genome_v2 \
        --out-md tmp/scripts/autopilot/milestones/lr-sweep-results.md
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
from glob import glob
from pathlib import Path

LN4 = math.log(4)  # 1.3863

NANOGPT_VAL_RE = re.compile(r"(?<![a-z_])val[_ ]loss\s+([\d.]+|nan)", re.IGNORECASE)
HF_EVAL_RE = re.compile(r"['\"]eval_loss['\"]\s*:\s*([\d.]+|nan)", re.IGNORECASE)


def extract_val_losses(log_path: Path) -> list[float]:
    text = log_path.read_text(errors="replace")
    losses: list[float] = []
    for line in text.splitlines():
        m = NANOGPT_VAL_RE.search(line) or HF_EVAL_RE.search(line)
        if m:
            v = m.group(1)
            losses.append(float("nan") if v.lower() == "nan" else float(v))
    return losses


def classify(losses: list[float], warmup: int = 2) -> tuple[str, str]:
    """Return (verdict, detail). Verdict in
    {PASS, COLLAPSE, SPIKE, NAN, INCOMPLETE, MARGINAL}."""
    if any(v != v for v in losses):
        return ("NAN", "NaN detected in val loss")
    post = losses[warmup:]
    if len(post) < 3:
        return ("INCOMPLETE", f"only {len(losses)} evals total, need ≥ {warmup+3}")

    lo, hi = min(post), max(post)
    last = post[-1]

    # SPIKE: max after warmup > min * 1.3 AND last is not the min
    if lo > 0 and hi / lo > 1.3 and last > lo * 1.15:
        return ("SPIKE", f"post-warmup max/min = {hi:.4f}/{lo:.4f} = {hi/lo:.2f}, last={last:.4f}")

    # COLLAPSE: last N mostly flat near ln(4)
    window = post[-min(5, len(post)):]
    if len(window) >= 3:
        span = max(window) - min(window)
        near = all(abs(v - LN4) <= 0.15 for v in window)
        if near and span <= 0.05:
            return ("COLLAPSE",
                    f"last {len(window)} evals [{min(window):.4f}..{max(window):.4f}] "
                    f"near ln(4)={LN4:.4f}, span {span:.4f}")

    # PASS: loss ended clearly below ln(4)/2 and monotone-ish descent
    # (charter: "half a step below ln(4)" → interpret as at least 20% below)
    if last <= LN4 * 0.80 and last < lo * 1.05:  # last is at/near the min
        return ("PASS", f"last={last:.4f} < 0.8·ln(4)={LN4*0.8:.4f}, min={lo:.4f}")

    # Otherwise MARGINAL — descending but not clearly below ln(4)/2
    return ("MARGINAL",
            f"last={last:.4f} (ln(4)={LN4:.4f}, min={lo:.4f}); descent unclear or slow")


def find_log(learning_source: Path, arch: str, subset: str, tag: str) -> Path | None:
    """Log file naming (see workflows/03[ac]-genome_sequence-train-*-small-subset.sh):
      <subset>-<arch>-small[-<TAG>]-YYYY-MM-DD_HH-MM-SS.log"""
    pattern = str(learning_source / "genome_sequence" / "logs" /
                  f"{subset}-{arch}-small-{tag}-*.log")
    matches = sorted(glob(pattern))
    if matches:
        return Path(matches[-1])
    # fallback: run without TAG (shouldn't happen for sweep)
    pattern = str(learning_source / "genome_sequence" / "logs" /
                  f"{subset}-{arch}-small-*.log")
    matches = sorted(glob(pattern))
    return Path(matches[-1]) if matches else None


def sacct_state(jobid: str) -> str:
    try:
        r = subprocess.run(["sacct", "-j", jobid, "-n", "-o", "State", "-P", "-X"],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return "UNKNOWN"
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return lines[0].split()[0] if lines else "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", required=True)
    ap.add_argument("--learning-source", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    lsd = Path(args.learning_source)
    runs = []
    for line in Path(args.roster).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        jobid, arch, subset, tag, lr_value = parts[:5]
        runs.append({
            "jobid": jobid, "arch": arch, "subset": subset,
            "tag": tag, "lr": float(lr_value),
        })

    for r in runs:
        r["slurm_state"] = sacct_state(r["jobid"])
        log = find_log(lsd, r["arch"], r["subset"], r["tag"])
        r["log_path"] = str(log) if log else "(not found)"
        if log is None:
            r["verdict"], r["detail"] = "INCOMPLETE", "no log file"
            r["losses"] = []
        else:
            losses = extract_val_losses(log)
            r["losses"] = losses
            r["verdict"], r["detail"] = classify(losses)

    # Pick largest PASSing LR per (arch, subset)
    winners: dict[tuple[str, str], dict] = {}
    for r in runs:
        if r["verdict"] != "PASS":
            continue
        key = (r["arch"], r["subset"])
        cur = winners.get(key)
        if cur is None or r["lr"] > cur["lr"]:
            winners[key] = r

    # Cross-subset winner per arch (both subsets must PASS at that LR)
    arch_winners: dict[str, tuple[float, str]] = {}
    for arch in ("bert", "gpt2"):
        # For each LR value, check if both subsets PASSed at that LR
        by_lr: dict[float, list[dict]] = {}
        for r in runs:
            if r["arch"] != arch:
                continue
            by_lr.setdefault(r["lr"], []).append(r)
        best_lr = None
        for lr, group in sorted(by_lr.items(), reverse=True):
            passes = [g for g in group if g["verdict"] == "PASS"]
            if len(passes) == len(group) and len(passes) >= 2:  # both subsets PASS
                best_lr = lr
                break
        if best_lr is not None:
            arch_winners[arch] = (best_lr, "both subsets PASS")

    lines = [
        "# genome subset small — mini LR sweep 結果",
        "",
        f"作成: {subprocess.run(['date', '-Iseconds'], capture_output=True, text=True).stdout.strip()}",
        f"charter 2026-07-14 reply §「mini LR sweep」対応",
        f"合格条件: loss が **ln(4)≒{LN4:.4f} の半歩下** で滑らかに下がる (last < 0.8·ln(4))、",
        f"上への spike なし、 collapse (ln(4) 近辺で平ら) なし。 合格の中で **arch ごとに最大安定 LR** を採用。",
        "",
        "## 個別 run 結果",
        "",
        "| jobid | arch | subset | LR | SLURM | 判定 | last val | detail |",
        "|---|---|---|---:|---|---|---:|---|",
    ]
    for r in sorted(runs, key=lambda x: (x["arch"], x["subset"], -x["lr"])):
        last = f"{r['losses'][-1]:.4f}" if r["losses"] else "-"
        lines.append(
            f"| {r['jobid']} | {r['arch']} | {r['subset']} | {r['lr']} | "
            f"{r['slurm_state']} | **{r['verdict']}** | {last} | {r['detail']} |"
        )

    lines += ["", "## arch ごとの採用 LR 候補", ""]
    if arch_winners:
        for arch, (lr, reason) in arch_winners.items():
            lines.append(f"- **{arch}**: LR = **{lr}** ({reason})")
    else:
        lines.append("- **どの arch も両 subset で PASS する LR が無い** — 追加 sweep が必要 (更に低い LR 帯)")

    lines += [
        "",
        "## 個別 subset で PASS したもの (参考)",
        "",
    ]
    if winners:
        for (arch, subset), r in sorted(winners.items()):
            lines.append(f"- {arch} / {subset}: max PASS LR = **{r['lr']}** (jobid {r['jobid']})")
    else:
        lines.append("- なし")

    lines += [
        "",
        "## val loss 曲線 (各 run、 全 eval)",
        "",
    ]
    for r in sorted(runs, key=lambda x: (x["arch"], x["subset"], -x["lr"])):
        if not r["losses"]:
            continue
        curve = ", ".join(f"{v:.4f}" for v in r["losses"])
        lines.append(f"- **{r['arch']} / {r['subset']} / LR {r['lr']}** ({len(r['losses'])} evals): {curve}")

    lines += [
        "",
        "## 次アクション",
        "",
        "1. 上長が本 report を review",
        "2. 採用 LR を config template のデフォルトに反映 (config commit) — 必要なら",
        "3. `subset_training.enabled = true` を立てて 42 本 launch (charter § 順守)",
        "",
    ]

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"[OK] wrote {args.out_md}")
    print(f"     arch winners: {arch_winners}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
