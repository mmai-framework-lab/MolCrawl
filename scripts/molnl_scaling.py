#!/usr/bin/env python3
"""Loss against model size at each learning rate: does a power law hold here?

A scaling law is a claim that crosses model sizes, so the x axis is compute,
C = 6ND, with N the NON-embedding parameter count (run-completion-figures, "Which x
axis"). Every arm of one grid reads the same number of tokens, so C is proportional to N
within a grid and the fitted exponent is the same either way -- this is a fixed-data
slice through L(N, D), not the compute-optimal frontier.

What that distinction costs here: molecule_nat_lang holds 325.8M training tokens, and a
1,500-iteration GPT-2 run reads 3.93G of them, so every size sees the same data 12 times
over. A law fitted on that slice says how far a bigger model gets on THIS corpus, not how
it would scale with data to match.

N is counted from each checkpoint, not from the config: the embedding table is 39M
parameters at this vocabulary, which is a third of small and would put molnl small
alongside models that cost the same to train but carry a smaller vocabulary.

    python scripts/molnl_scaling.py --bert-root <dir> --gpt2-root <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import japanize_matplotlib                           # noqa: E402,F401
import matplotlib.pyplot as plt                      # noqa: E402

INK, INK_2, INK_3 = "#0f161a", "#53626c", "#8695a0"
SURFACE, RULE = "#fcfcfb", "#d7e0e5"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARKERS = {"small": "o", "medium": "s", "large": "^", "xl": "D"}
SIZE_ORDER = ("small", "medium", "large", "xl")

# Tensors that are lookup tables rather than matrix multiplies. 6ND counts the latter.
BERT_EMBED = ("embeddings.word_embeddings", "embeddings.position_embeddings",
              "embeddings.token_type_embeddings", "cls.predictions.decoder")
GPT2_EMBED = ("wte.weight", "wpe.weight", "lm_head.weight")
TOKENS_PER_STEP = 2560 * 1024


def safetensors_shapes(path):
    """{name: shape} from the header alone. No weights are read."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}


def torch_shapes(path):
    """{name: shape} from a torch .bin, memory-mapped so the weights stay on disk."""
    import torch

    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    return {k: list(v.shape) for k, v in state.items()}


def non_embedding(shapes, embed_keys):
    total = 0
    for name, shape in shapes.items():
        if any(k in name for k in embed_keys):
            continue
        n = 1
        for dim in shape:
            n *= dim
        total += n
    return total


def newest_checkpoint(run_dir):
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[1]))
    return cks[-1] if cks else None


def bert_arms(root, pattern, last_n=10, exclude=None):
    """One row per BERT arm: size, rate, non-embedding N, window mean, state."""
    name_re = re.compile(r"^bert_(?P<size>small|medium|large)_(?:(?P<sched>\w+?)_)?lr(?P<tag>[0-9p]+e\d)$")
    arms, params = [], {}
    for run_dir in sorted(glob.glob(os.path.join(root, pattern))):
        if exclude and exclude in os.path.basename(run_dir):
            continue
        m = name_re.match(os.path.basename(run_dir))
        ck = newest_checkpoint(run_dir)
        if not m or not ck:
            continue
        state = json.load(open(os.path.join(ck, "trainer_state.json")))
        pts = sorted((e["step"], e["eval_loss_mask"]) for e in state["log_history"]
                     if "eval_loss_mask" in e)
        if not pts:
            continue
        size = m["size"]
        if size not in params:
            sf = os.path.join(ck, "model.safetensors")
            params[size] = non_embedding(safetensors_shapes(sf), BERT_EMBED) if os.path.exists(sf) else None
        tail = [v for _, v in pts[-last_n:]]
        arms.append({
            "size": size, "lr": float(m["tag"].replace("p", ".").replace("e", "e-")),
            "params": params[size], "window": st.mean(tail), "sd": st.pstdev(tail),
            "last_step": pts[-1][0], "max_steps": int(state.get("max_steps") or 0),
            "tokens": pts[-1][0] * TOKENS_PER_STEP,
            "done": pts[-1][0] >= int(state.get("max_steps") or 0),
        })
    return arms


def gpt2_arms(root, iters=1500, every=50, last_n=10):
    name_re = re.compile(r"^gpt2_(?P<size>small|medium|large|xl)_(?P<iters>\d+)_lr(?P<tag>[0-9p]+e\d)$")
    arms, params = [], {}
    for run_dir in sorted(glob.glob(os.path.join(root, "gpt2_*_lr*"))):
        m = name_re.match(os.path.basename(run_dir))
        if not m or int(m["iters"]) != iters:
            continue
        rows = []
        for path in glob.glob(os.path.join(run_dir, "logging_*.csv")):
            got = [r for r in csv.reader(open(path)) if r and r[0].strip().isdigit()]
            if len(got) > len(rows):
                rows = got
        if not rows:
            continue
        steps = [int(r[0]) for r in rows]
        vals = [float(r[2]) for r in rows]
        size = m["size"]
        if size not in params:
            ck = newest_checkpoint(run_dir)
            bin_path = os.path.join(ck, "pytorch_model.bin") if ck else None
            params[size] = (non_embedding(torch_shapes(bin_path), GPT2_EMBED)
                            if bin_path and os.path.exists(bin_path) else None)
        start = iters - (last_n - 1) * every
        tail = [v for s, v in zip(steps, vals) if s >= start]
        arms.append({
            "size": size, "lr": float(m["tag"].replace("p", ".").replace("e", "e-")),
            "params": params[size], "window": st.mean(tail) if tail else None,
            "sd": st.pstdev(tail) if len(tail) > 1 else 0.0,
            "last_step": steps[-1], "max_steps": iters,
            "tokens": steps[-1] * TOKENS_PER_STEP, "done": steps[-1] >= iters,
        })
    return arms


def fit(points):
    """Least-squares slope of log(loss) on log(C), and the fit's own residual."""
    xs = [math.log10(x) for x, _ in points]
    ys = [math.log10(y) for _, y in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if not den:
        return None, None, None
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    b = my - a * mx
    resid = max(abs(y - (a * x + b)) for x, y in zip(xs, ys))
    return a, b, resid


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
        ax.spines[s].set_linewidth(.8)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3, width=.8)
    ax.grid(color=RULE, linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def finish(fig, title, caption, path):
    import textwrap
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=INK, x=.012, ha="left", y=.985)
    wrapped = "\n".join(textwrap.wrap(caption, width=132, break_long_words=True))
    fig.subplots_adjust(bottom=max(.20, .10 + .035 * (wrapped.count("\n") + 1)))
    fig.text(.012, .012, wrapped, fontsize=7.6, color=INK_3, ha="left", va="bottom",
             linespacing=1.6)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def envelope(arms):
    """Each size at whichever rate did best for it.

    A line at one learning rate crosses sizes that are not equally well tuned: the rate
    that suits small is already past the point where large diverges. The envelope is the
    comparison a scaling law is about.
    """
    out = []
    for size in SIZE_ORDER:
        same = [a for a in arms if a["size"] == size and a["window"] is not None and a["params"]]
        if same:
            out.append(min(same, key=lambda a: a["window"]))
    return out


def draw(arms, title, caption, ylabel, path, floor=None, floor_label=""):
    """One line per learning rate: loss against compute, sizes along the line."""
    # A rate that only one size ran carries no size comparison, and there are more of
    # them than the palette has colours. They are drawn -- the envelope runs through
    # them -- but in one grey, with one legend entry.
    by_rate = {}
    for a in arms:
        if a["window"] is not None and a["params"]:
            by_rate.setdefault(a["lr"], []).append(a)
    spanning = sorted(lr for lr, v in by_rate.items() if len(v) >= 2)
    single = sorted(lr for lr, v in by_rate.items() if len(v) == 1)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fits = []

    def mark(arm, colour):
        ax.plot([float(6 * arm["params"] * arm["tokens"])], [arm["window"]],
                marker=MARKERS[arm["size"]], markersize=8, color=colour,
                markerfacecolor=colour if arm["done"] else SURFACE,
                markeredgecolor=colour, markeredgewidth=1.6, zorder=4)

    for i, lr in enumerate(spanning):
        pts = sorted(by_rate[lr], key=lambda a: a["params"])
        usable = [(float(6 * a["params"] * a["tokens"]), a["window"]) for a in pts]
        colour = SERIES[i % len(SERIES)]
        ax.plot([p[0] for p in usable], [p[1] for p in usable], color=colour,
                linewidth=1.6, zorder=3, label=f"lr {lr:g}（{len(usable)} サイズ）")
        a_, b_, resid = fit(usable)
        fits.append((lr, a_, resid, len(usable)))
        for arm in pts:
            mark(arm, colour)
    # Three of large's four rates sit within 0.005 of each other: labels at one offset
    # land on top of one another.
    seen_at_size = {}
    for lr in single:
        arm = by_rate[lr][0]
        mark(arm, INK_3)
        k = seen_at_size.get(arm["size"], 0)
        seen_at_size[arm["size"]] = k + 1
        ax.annotate(f"{lr:g}", (float(6 * arm["params"] * arm["tokens"]), arm["window"]),
                    textcoords="offset points", xytext=(9, [-3, 7, -13, 15][k % 4]),
                    fontsize=7.4, color=INK_3)
    if single:
        ax.plot([], [], marker="o", color=INK_3, linestyle="none",
                label="そのサイズだけで走らせた学習率")
    env = envelope(arms)
    if len(env) >= 2:
        xs = [float(6 * a["params"] * a["tokens"]) for a in env]
        ys = [a["window"] for a in env]
        ax.plot(xs, ys, color=INK, linewidth=2.4, alpha=.28, zorder=2,
                label="各サイズの最良（包絡）")
        a_, b_, resid = fit(list(zip(xs, ys)))
        fits.append(("envelope", a_, resid, len(xs)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    values = [a["window"] for a in arms if a["window"] is not None and a["params"]]
    if values:
        lo, hi = min(values), max(values)
        span = math.log10(hi) - math.log10(lo)
        ax.set_ylim(10 ** (math.log10(lo) - span * .18), 10 ** (math.log10(hi) + span * .18))
    # One tick per size, labelled with the parameter count that sets it: a log axis
    # between 5e17 and 3e19 otherwise prints a single decade mark.
    by_size = {}
    for arm in arms:
        if arm["params"] and arm["window"] is not None:
            by_size[arm["size"]] = float(6 * arm["params"] * arm["tokens"])
    ticks = [(by_size[s_], s_) for s_ in SIZE_ORDER if s_ in by_size]
    if ticks:
        ax.set_xticks([float(t) for t, _ in ticks], minor=False)
        ax.set_xticklabels([f"{name}\n{[a for a in arms if a['size'] == name][0]['params'] / 1e6:.0f}M"
                            for _, name in ticks])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlim(float(ticks[0][0]) / 1.9, float(ticks[-1][0]) * 1.9)
    style(ax, "学習に使った計算量 C = 6ND（N は非埋め込みパラメータ数、目盛りはサイズ）", ylabel)
    for size in SIZE_ORDER:
        if any(a["size"] == size for a in arms):
            ax.plot([], [], marker=MARKERS[size], color=INK_3, linestyle="none", label=size)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=2, loc="lower left")
    finish(fig, title, caption, path)
    return fits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bert-root", required=True)
    ap.add_argument("--gpt2-root", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    sets = [
        ("bert24k", bert_arms(a.bert_root, "bert_*_24k_lr*"), "eval_loss_mask（nats/token）", 3.8638,
         "何も学ばない場合"),
        ("bert12k", bert_arms(a.bert_root, "bert_*_lr*", exclude="_24k_"),
         "eval_loss_mask（nats/token）", 3.8638, "何も学ばない場合"),
        ("gpt2", gpt2_arms(a.gpt2_root), "val loss（nats/token）", 4.6514, "何も学ばない場合"),
    ]
    rows = []
    for key, arms, ylabel, floor, floor_label in sets:
        healthy = [x for x in arms if x["window"] is not None and x["window"] < floor]
        for x in healthy:
            rows.append(dict(grid=key, **x))
        print(f"\n=== {key}: {len(arms)} arms, {len(healthy)} below the floor")
        for x in sorted(healthy, key=lambda r: (r["lr"], SIZE_ORDER.index(r["size"]))):
            where = "完走" if x["done"] else f"走行中 {x['last_step']:,}/{x['max_steps']:,}"
            print(f"    {x['size']:<7}lr={x['lr']:<9g}N={x['params'] or 0:>12,}  "
                  f"window={x['window']:.4f} sd={x['sd']:.4f} {where}")
    named = {k: v for k, (v, *_rest) in
             {k: (arms,) for k, arms, *_ in sets}.items()}
    base = ("前提: グローバルバッチ 2,560 系列（2.62M トークン/step）/ 訓練データ 325.8M トークン"
            "（318,118 系列 × 1,024）。横軸は C = 6ND、N は checkpoint から数えた非埋め込みパラメータ数。"
            "6ND は系列長由来の注意項（系列長 1,024 で 13〜18%）を含まない。1 つの格子の中では全 arm が"
            "同じトークン数を読むので C は N に比例し、これは L(N, D) の D 固定断面であって計算量最適の"
            "フロンティアではない。seed は各 arm 1 本で seed 間のばらつきは未測定 —— 順位として読まないこと。"
            "白抜きの印は走行中の run。")
    bert_caption = (base + " max_steps 24,000 / 193.1 epoch / 学習トークン 62.9G（完走時）。"
                    "縦軸は eval_loss_mask（[MASK] 位置のみを採点）。何も学ばないモデルの 3.8638 は軸の外（どの点もその 45〜65 分の 1）。"
                    "色は学習率を表すが、格子ごとに学習率の組が違うので BERT の図と GPT-2 の図で同じ色は同じ値ではない。"
                    "発散した 3 本（small 1.3e-3・medium 5.5e-4・medium 7.4e-4）はこの図に含めない。")
    gpt2_caption = (base + " max_iters 1,500 / 12.07 epoch / 学習トークン 3.93G。"
                    "縦軸は系列全体の次トークン損失。BERT と同じ軸には載せない（採点対象が違う）。"
                    "何も学ばないモデルの 4.6514 は軸の外（どの点もその 6〜8 分の 1）。"
                    "色は学習率を表すが、格子ごとに学習率の組が違うので BERT の図と同じ色は同じ値ではない。")
    fits_all = {}
    below = lambda arms_, floor_: [x for x in arms_ if x["window"] is not None and x["window"] < floor_]
    fits_all["bert24k"] = draw(
        below(named["bert24k"], 3.8638),
        "molecule_nat_lang BERT 24,000 step —— 同じ学習率でのサイズ比較と、各サイズ最良の包絡",
        bert_caption, "eval_loss_mask（nats/token）",
        os.path.join(a.out_dir, "scaling-bert-24k.png"), 3.8638, "何も学ばない場合")
    fits_all["gpt2"] = draw(
        below(named["gpt2"], 4.6514),
        "molecule_nat_lang GPT-2 1,500 iteration —— 同じ学習率でのサイズ比較と、各サイズ最良の包絡",
        gpt2_caption, "val loss（nats/token）",
        os.path.join(a.out_dir, "scaling-gpt2.png"), 4.6514, "何も学ばない場合")
    print("\n=== 傾き（log-log 最小二乗、L ∝ C^a）===")
    for key, fits in fits_all.items():
        for lr, slope, resid, n in fits:
            label = "包絡" if lr == "envelope" else f"lr {lr:g}"
            print(f"    {key:<8}{label:<14}a={slope:+.3f}  点数={n}  "
                  f"最大残差={resid:.3f} dex")

    with open(os.path.join(a.out_dir, "molnl-scaling-arms.tsv"), "w") as fh:
        cols = ["grid", "size", "lr", "params", "tokens", "window", "sd", "last_step",
                "max_steps", "done"]
        fh.write("# N is the non-embedding parameter count, from the checkpoint\n")
        fh.write("# window is the mean of the last 10 evaluations; sd its spread\n")
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(int(r[c]))
                               if isinstance(r[c], bool) else str(r[c]) for c in cols) + "\n")
    print(f"  wrote molnl-scaling-arms.tsv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
