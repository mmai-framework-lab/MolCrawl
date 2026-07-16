"""Parse a nanoGPT/HF training log for divergence signals.

charter §「本番 run が発散(NaN / val loss 上昇)→ その run だけ停止(フラグ、best checkpoint 採用、
他は続行)」対応。

Divergence signals:
1. NaN loss in any recent eval
2. val loss last-N mean > first-N mean × 1.3 (monotone worsening after warmup)
3. train loss NaN

Usage:
    python3 parse_train_log_for_divergence.py <log_file>
    → exit 0 = healthy, exit 1 = diverging (caller should park + kill)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


NAN_RE = re.compile(r"\bnan\b", re.IGNORECASE)
VAL_LOSS_RE = re.compile(r"(?:val|eval)[_ ]?loss\s*[=:]\s*([\d.]+|nan)", re.IGNORECASE)
TRAIN_LOSS_RE = re.compile(r"train[_ ]?loss\s*[=:]\s*([\d.]+|nan)", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_file")
    ap.add_argument("--warmup-evals", type=int, default=2,
                    help="ignore first N val-loss points (still warming up)")
    ap.add_argument("--recent-window", type=int, default=3,
                    help="compare last-N val loss mean vs first-N post-warmup mean")
    ap.add_argument("--degrade-ratio", type=float, default=1.30,
                    help="last mean / first mean > this = divergence")
    args = ap.parse_args()

    text = Path(args.log_file).read_text(errors="replace")

    val_losses: list[float] = []
    train_losses: list[float] = []
    for line in text.splitlines():
        m = VAL_LOSS_RE.search(line)
        if m:
            v = m.group(1)
            val_losses.append(float("nan") if v.lower() == "nan" else float(v))
        m = TRAIN_LOSS_RE.search(line)
        if m:
            v = m.group(1)
            train_losses.append(float("nan") if v.lower() == "nan" else float(v))

    reasons: list[str] = []
    # NaN in either loss
    if any(v != v for v in val_losses):
        reasons.append("val_loss NaN detected")
    if any(v != v for v in train_losses):
        reasons.append("train_loss NaN detected")

    # Val loss trend
    post_warmup = val_losses[args.warmup_evals:]
    if len(post_warmup) >= 2 * args.recent_window:
        first_mean = sum(post_warmup[: args.recent_window]) / args.recent_window
        last_mean = sum(post_warmup[-args.recent_window:]) / args.recent_window
        if first_mean > 0 and last_mean / first_mean > args.degrade_ratio:
            reasons.append(
                f"val_loss degrading: first_{args.recent_window}={first_mean:.4f} → "
                f"last_{args.recent_window}={last_mean:.4f} (ratio {last_mean/first_mean:.2f})"
            )

    if reasons:
        print("[DIVERGENCE]", "; ".join(reasons))
        return 1
    print(f"[HEALTHY] {len(val_losses)} val samples, {len(train_losses)} train samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
