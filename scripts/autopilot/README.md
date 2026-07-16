# autopilot — 4 day autonomous driver

作成: 2026-07-10
根拠: `tmp/docs_tmp_local/yigarashi-issue/instruction-4day-autonomous-charter.md`

## 概要

4 日間サーバに繋げられない状態で、 genome G2 → config 監査 → LR check → smoke → 本番学習を
SLURM 上で自動進行させるための driver 群。

## ディレクトリ

- `sbatch/` — SLURM sbatch template (compounds train, genome G2 各 step)
- `analyzers/` — 中間解析 (LR 発散検知、 G2 realized 窓集計 → target 決定)
- `state/` — driver state (JSON、 進行状況 / SLURM JOBID / park フラグ)
- `logs/` — driver 自身の log
- `milestones/` — 完了 milestone md (compounds/subset/mammal 各 size)

## 自律判断ルール (matsubara の代役、 charter §「自律判断ルール」より)

1. realized 窓数に外れ値 → 中央値近傍で target 決定 (最小値には引きずられない、 seed6 型を回避)
2. LR — 発散したら park、 本番 run で早期 val_loss 悪化検知 → best ckpt 採用 + park
3. max_iters — 実測窓数から `ceil(3 × 窓数 / 2560)` で自動計算
4. global batch — per_device が小さい config は grad_accum を上げて 2,560 に (Phase 1-4 で compounds 反映済)
5. smoke で落ちる config → skip + park
6. 本番 run 発散 → その run のみ停止、 best ckpt 採用、 他 modality 続行
7. 人判断が要る 1 件 → `park_log.md` に追記して他は進める

## ガードレール

- 出力は新ディレクトリ (`learning_source_20260710_*_v2/`)、 既存 world 無傷
- PR マージなし、 branch/worktree のまま
- HF Hub の archive/delete なし
- OpenGenome2 全データ学習は開始しない

## 実行

`bash tmp/scripts/autopilot/kickoff.sh` で launch (nohup で coordinator を上げる + 初期 SLURM 投入)。
