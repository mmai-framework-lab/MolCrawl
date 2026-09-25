"""終わった checkpoint を、固定した検証集合で測り直す。学習はしない。

格子は評価のたびに検証行を引き直しており、その散らばりが 0.010-0.012 ある。
読みたい動きはそれより小さいので、同じ行を全点に読ませて取り直す。

損失の定義は train.py の get_batch と estimate_loss をそのまま写している。
block_size まで 0 で詰め、x を 1 つ手前まで、y を 1 つ先からにして model(X, Y)
の返す損失を取り、バッチごとの平均をさらに平均する。詰めた位置を損失から外す
かどうかも config の指定に従う。定義がずれると、取り直した値と格子の値が
別のものになり、並べた表が意味を失う。
"""
import argparse
import csv
import glob
import importlib
import os
import re

import numpy as np
import torch

RUN_RE = re.compile(r"-(small|medium|large|xl)-lr(\w+)$")
LR_NUM = {"6e4": 6e-4, "3e4": 3e-4, "1p5e4": 1.5e-4, "1p2e3": 1.2e-3,
          "1e4": 1e-4, "1e3": 1e-3, "7p5e5": 7.5e-5}


def strip_compile_prefix(state):
    pre = "_orig_mod."
    return {(k[len(pre):] if k.startswith(pre) else k): v for k, v in state.items()}


def load_model(path, device):
    from molcrawl.models.gpt2.model import GPT, GPTConfig
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**dict(ck["model_args"])))
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    model.to(device).eval()
    return model, int(ck.get("iter_num", -1))


def make_batch(data, ix, block_size):
    """train.py の get_batch と同じ詰め方で 1 バッチ作る。"""
    rows = []
    for i in ix:
        seq = data[i]
        seq = seq.long() if seq.dtype != torch.long else seq
        if len(seq) > block_size:
            rows.append(seq[:block_size])
        elif len(seq) < block_size:
            rows.append(torch.cat([seq, torch.zeros(block_size - len(seq), dtype=torch.long)]))
        else:
            rows.append(seq)
    batch = torch.stack(rows)
    return batch[:, :-1].long(), batch[:, 1:].long()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="測る点の config。データの場所をここから取る")
    ap.add_argument("--runs-root", required=True, help="格子の出力の根")
    ap.add_argument("--out", required=True, help="書き出す TSV")
    ap.add_argument("--rows", type=int, default=0,
                    help="検証集合の行数。0 なら valid 全体。先頭から取る")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--checkpoints", default="ckpt.pt,checkpoint-40320/training_state.bin",
                    help="各 run の中で読むファイル。コンマ区切りで複数")
    ap.add_argument("--only", default="", help="この文字列を含む run だけ測る")
    a = ap.parse_args(argv)

    cfg = importlib.import_module(a.config.replace("/", ".").removesuffix(".py"))
    block_size = cfg.block_size
    ambiguous = list(getattr(cfg, "ambiguous_token_ids", []) or [])
    pad_for_loss = getattr(cfg, "pad_token_id_for_loss", None)

    from molcrawl.data.rna.dataset.rna_dataset import RNABinDataset
    data = RNABinDataset(cfg.rna_bin_dir, split="valid", vocab_file=cfg.rna_vocab_file)
    n_rows = len(data) if a.rows <= 0 else min(a.rows, len(data))
    ix_all = list(range(n_rows))  # 先頭から。どの点も同じ行を同じ順で読む
    print(f"  valid {len(data):,} 行のうち {n_rows:,} 行を使う  block_size {block_size}")
    print(f"  詰めた位置を損失から外す: {pad_for_loss}  外す token: {ambiguous or 'なし'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device {device}")

    wanted = [c.strip() for c in a.checkpoints.split(",") if c.strip()]
    runs = []
    for d in sorted(glob.glob(f"{a.runs_root}/*")):
        m = RUN_RE.search(os.path.basename(d))
        if not m or (a.only and a.only not in os.path.basename(d)):
            continue
        for name in wanted:
            path = os.path.join(d, name)
            if not os.path.exists(path):
                print(f"  SKIP {os.path.basename(d)}: {name} が無い")
                continue
            runs.append((m.group(1), m.group(2), path))
    if not runs:
        raise SystemExit(f"{a.runs_root} の下に測れる run が無い")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["arch", "size", "learning_rate", "step", "val_loss_fixed",
                    "rows", "batches", "checkpoint"])
        for size, lr, path in runs:
            model, step = load_model(path, device)
            losses = []
            with torch.no_grad():
                for s in range(0, n_rows - a.batch_size + 1, a.batch_size):
                    x, y = make_batch(data, ix_all[s:s + a.batch_size], block_size)
                    if ambiguous or pad_for_loss is not None:
                        from molcrawl.models._collators import mask_ambiguous_targets_for_clm
                        if ambiguous:
                            y = mask_ambiguous_targets_for_clm(y, ambiguous)
                        if pad_for_loss is not None:
                            y = mask_ambiguous_targets_for_clm(y, [pad_for_loss])
                    _logits, loss = model(x.to(device), y.to(device))
                    losses.append(loss.item())
            mean = float(np.mean(losses))
            w.writerow(["gpt2", size, LR_NUM.get(lr, lr), step, f"{mean:.6f}",
                        n_rows, len(losses), os.path.basename(os.path.dirname(path))
                        or os.path.basename(path)])
            fh.flush()
            print(f"  {size:<7} lr{lr:<7} step {step:>7,}  "
                  f"固定検証 {mean:.4f}  バッチ {len(losses):,}")
            del model
            torch.cuda.empty_cache()
    print(f"  WROTE {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
