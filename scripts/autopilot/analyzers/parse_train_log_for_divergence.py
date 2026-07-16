"""Parse a nanoGPT/HF training log for divergence signals.

charter §「本番 run が発散(NaN / val loss 上昇)→ その run だけ停止(フラグ、best checkpoint 採用、
他は続行)」対応。 2026-07-14 更新: SLURM COMPLETED 後にも呼ぶ用途で
`--last-vs-best` モードを追加 — 最終 val loss が best より一定比 (default 1.3×) 悪化して
いれば diverged と判定 (bert-large タイプの見逃しを塞ぐ)。

Divergence signals:
1. NaN loss in any eval / train sample
2. val loss last-N mean > first-N mean × ratio (monotone worsening after warmup)
3. (new) --last-vs-best mode: final val loss / best val loss > ratio

Log formats understood:
- nanoGPT:      `step 12000: train loss 0.16, val loss 0.17`
- HF Trainer:   `'eval_loss': 0.1745`
- NaN token in either

Usage:
    # Realtime (used during running / on FAILED):
    python3 parse_train_log_for_divergence.py <log_file>

    # Post-COMPLETED (last-vs-best):
    python3 parse_train_log_for_divergence.py <log_file> --last-vs-best

Exit codes: 0 = healthy, 1 = diverging (caller should park).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


NAN_RE = re.compile(r"\bnan\b", re.IGNORECASE)
# nanoGPT: "val loss 0.17" or "val_loss: 0.17"
NANOGPT_VAL_RE = re.compile(r"(?<![a-z_])val[_ ]loss\s+([\d.]+|nan)", re.IGNORECASE)
# HF Trainer: "'eval_loss': 0.1745"
HF_EVAL_RE = re.compile(r"['\"]eval_loss['\"]\s*:\s*([\d.]+|nan)", re.IGNORECASE)
# nanoGPT train loss
NANOGPT_TRAIN_RE = re.compile(r"train[_ ]loss\s+([\d.]+|nan)", re.IGNORECASE)


def _extract_losses(text: str) -> tuple[list[float], list[float]]:
    """Return (val_losses, train_losses) from a training log; format-agnostic."""
    val_losses: list[float] = []
    train_losses: list[float] = []
    for line in text.splitlines():
        m = NANOGPT_VAL_RE.search(line) or HF_EVAL_RE.search(line)
        if m:
            v = m.group(1)
            val_losses.append(float("nan") if v.lower() == "nan" else float(v))
        m = NANOGPT_TRAIN_RE.search(line)
        if m:
            v = m.group(1)
            train_losses.append(float("nan") if v.lower() == "nan" else float(v))
    return val_losses, train_losses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_file")
    ap.add_argument("--warmup-evals", type=int, default=2,
                    help="ignore first N val-loss points (still warming up)")
    ap.add_argument("--recent-window", type=int, default=3,
                    help="compare last-N val loss mean vs first-N post-warmup mean")
    ap.add_argument("--degrade-ratio", type=float, default=1.30,
                    help="last mean / first mean > this = divergence")
    ap.add_argument("--last-vs-best", action="store_true",
                    help="also fire when final val loss / min(post-warmup val loss) > "
                         "degrade-ratio. Use this on COMPLETED runs to catch bert-large-"
                         "style divergence where the schedule ran to the end but val loss "
                         "walked away from its best value.")
    # charter 2026-07-14 reply §「collapse 判定」: pre-G2 で BERT 1e-4 が
    # ln(4)≈1.386 に collapse する事象 (degenerate near-uniform output) を、
    # 「low & flat」 パターンとして検出する。 上向き発散だけを見る --last-vs-best
    # では見逃す (loss は低いから healthy に見える) → 縮退モデルの量産を防ぐ。
    ap.add_argument("--collapse-check", action="store_true",
                    help="also fire when the last --collapse-window val losses are ALL "
                         "within [collapse-target ± collapse-tolerance] AND vary by less "
                         "than collapse-flat-span across the window (i.e. loss is low "
                         "and flat near a plateau).")
    ap.add_argument("--collapse-window", type=int, default=5,
                    help="how many consecutive last evals to require flat (default 5)")
    ap.add_argument("--collapse-target", type=float, default=1.386,
                    help="collapse plateau value; default 1.386 ≈ ln(4) = uniform "
                         "prediction over ACGT (single-nt vocab).")
    ap.add_argument("--collapse-tolerance", type=float, default=0.15,
                    help="max |loss - target| for a sample to count as 'near plateau' "
                         "(default 0.15 → [1.236, 1.536] around ln(4)).")
    ap.add_argument("--collapse-flat-span", type=float, default=0.05,
                    help="max (max - min) across the collapse window for it to count "
                         "as flat (default 0.05).")
    args = ap.parse_args()

    text = Path(args.log_file).read_text(errors="replace")
    val_losses, train_losses = _extract_losses(text)

    reasons: list[str] = []
    # NaN in either loss
    if any(v != v for v in val_losses):
        reasons.append("val_loss NaN detected")
    if any(v != v for v in train_losses):
        reasons.append("train_loss NaN detected")

    # Val loss trend (first-N mean vs last-N mean after warmup)
    post_warmup = val_losses[args.warmup_evals:]
    if len(post_warmup) >= 2 * args.recent_window:
        first_mean = sum(post_warmup[: args.recent_window]) / args.recent_window
        last_mean = sum(post_warmup[-args.recent_window:]) / args.recent_window
        if first_mean > 0 and last_mean / first_mean > args.degrade_ratio:
            reasons.append(
                f"val_loss degrading: first_{args.recent_window}={first_mean:.4f} → "
                f"last_{args.recent_window}={last_mean:.4f} (ratio {last_mean/first_mean:.2f})"
            )

    # last-vs-best (post-COMPLETED bert-large-style detection).
    if args.last_vs_best and len(post_warmup) >= 3:
        best = min(v for v in post_warmup if v == v)  # skip NaN
        last = post_warmup[-1]
        if best > 0 and last / best > args.degrade_ratio:
            reasons.append(
                f"final vs best: last={last:.4f} vs best={best:.4f} "
                f"(ratio {last/best:.2f} > {args.degrade_ratio})"
            )

    # collapse: last N evals all near a plateau AND flat.
    if args.collapse_check and len(post_warmup) >= args.collapse_window:
        window = post_warmup[-args.collapse_window:]
        window = [v for v in window if v == v]  # skip NaN
        if len(window) == args.collapse_window:
            near = all(abs(v - args.collapse_target) <= args.collapse_tolerance for v in window)
            flat = (max(window) - min(window)) <= args.collapse_flat_span
            if near and flat:
                reasons.append(
                    f"collapse: last {args.collapse_window} evals "
                    f"= [{min(window):.4f}..{max(window):.4f}] near target "
                    f"{args.collapse_target} (span {max(window)-min(window):.4f} <= "
                    f"{args.collapse_flat_span}, |mean-target|={abs(sum(window)/len(window)-args.collapse_target):.3f} "
                    f"<= {args.collapse_tolerance})"
                )

    if reasons:
        print("[DIVERGENCE]", "; ".join(reasons))
        return 1
    print(f"[HEALTHY] {len(val_losses)} val samples, {len(train_losses)} train samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
