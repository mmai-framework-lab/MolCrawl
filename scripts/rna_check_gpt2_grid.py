"""格子 13 本の config を実行して、投入前の 7 項目を確かめる。

読むのではなく実行する。config は import * と環境変数を使うので、
ファイルを目で追っても実効値は分からない。
"""
import argparse
import glob
import io
import os
import sys

WANT_LR = {"6e4": 6e-4, "3e4": 3e-4, "1p5e4": 1.5e-4, "7p5e5": 7.5e-5}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config-dir", default="molcrawl/tasks/pretrain/configs/rna")
ap.add_argument("--pattern", default="gpt2_*_lr*.py")
A = ap.parse_args()
base = A.config_dir
files = sorted(glob.glob(f"{base}/{A.pattern}"))
print(f"  対象 {len(files)} 本\n")
hdr = f"  {'config':26s} {'MBxGA':>10s} {'x4GPU':>7s} {'decl':>6s} {'lr':>8s} {'seed':>5s} {'dtype':>9s}"
print(hdr)
outs, bad = {}, []
for f in files:
    g = {"__name__": "__main__", "__file__": f}
    try:
        exec(compile(io.open(f, encoding="utf-8").read(), f, "exec"), g)
    except SystemExit as e:
        bad.append((f, f"SystemExit: {e}"))
        continue
    except Exception as e:
        bad.append((f, f"{type(e).__name__}: {e}"))
        continue
    name = os.path.basename(f)[:-3]
    tag = name.rsplit("_lr", 1)[1]
    mb, ga = g["batch_size"], g["gradient_accumulation_steps"]
    eff = mb * ga
    lr, seed, dt = g["learning_rate"], g["seed"], g.get("dtype")
    decl = g.get("expected_global_batch")
    outs[name] = g["out_dir"]
    ok = (eff == 2560 and decl == 2560 and abs(lr - WANT_LR[tag]) < 1e-12
          and seed == 42 and dt == "bfloat16")
    print(f"  {'✅' if ok else '❌'} {name:23s} {mb:4d}x{ga:<5d} {eff:7d} {str(decl):>6s} "
          f"{lr:8.2e} {seed:5d} {str(dt):>9s}")
    if not ok:
        bad.append((f, f"eff={eff} decl={decl} lr={lr} seed={seed} dtype={dt}"))
print()
# 5: 出力先が互いに重ならない / 6: 既存 checkpoint が無い
dup = len(outs) - len(set(outs.values()))
print(f"  5 出力先の重複: {dup} 件")
print("  6 既存 checkpoint:")
for n, d in sorted(outs.items()):
    ck = sorted(glob.glob(os.path.join(d, "ckpt*.pt"))) + sorted(glob.glob(os.path.join(d, "checkpoint-*")))
    print(f"      {n:26s} {'空' if not ck else '❌ ' + str(len(ck)) + ' 件'}  {d}")
print()
if bad:
    print("  ❌ 問題:")
    for f, why in bad:
        print(f"      {os.path.basename(f)}: {why}")
sys.exit(1 if (bad or dup) else 0)
