#!/usr/bin/env python3
"""What the input form does to the retrieval score.

The first fourteen measurements asked in a form no model had seen in pretraining: joined
with a newline where the corpus writes <s>[INST] ... [/INST]...<|endoftext|>, and with the
molecule in SELFIES where the corpus writes SMILES. The 2026-10-23 judgment asked for the
two mismatches to be separated, so each model is measured four ways and drawn across them.

The x axis is the input form, not a quantity -- the four are ordered by how much of the
corpus's own shape they restore, and the lines between them are there to follow a model
across the four, not to suggest anything in between.

BERT and GPT-2 share these axes. A recall is a downstream metric and not a loss, so unlike
eval_loss_mask against val_loss the two are the same measurement here.

    python scripts/molnl_retrieval_format_figure.py --formats-dir <dir> --baseline-dir <dir> \
        --out-dir <dir> --caption <file>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401,E402  (registers a CJK font)
import matplotlib.pyplot as plt  # noqa: E402

from molnl_loss_figures import INK, SIZE_COLOUR, axes  # noqa: E402

FORMS = [("plain/selfies", "改行\nSELFIES"), ("inst/selfies", "マーカー\nSELFIES"),
         ("plain/smiles", "改行\nSMILES"), ("inst/smiles", "マーカー\nSMILES")]
MODELS = [("bert", "small"), ("bert", "medium"), ("bert", "large"),
          ("gpt2", "small"), ("gpt2", "medium"), ("gpt2", "large"), ("gpt2", "xl")]
DIRECTIONS = (("molecule_to_text", "分子 → 説明文"), ("text_to_molecule", "説明文 → 分子"))
CHANCE = {"recall@1": 0.01, "mrr": 0.0519}
STYLE = {"bert": "-", "gpt2": "--"}


def load(formats_dir, baseline_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(baseline_dir, "retrieval-*.json"))):
        d = json.load(open(path))
        for direction, v in d["directions"].items():
            out[((d["arch"], d["size"]), "plain/selfies", direction)] = v
    for path in sorted(glob.glob(os.path.join(formats_dir, "retrieval-*.json"))):
        d = json.load(open(path))
        form = f"{d['wrap']}/{d['molecule_notation']}"
        for direction, v in d["directions"].items():
            out[((d["arch"], d["size"]), form, direction)] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--formats-dir", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--metric", default="recall@1", choices=("recall@1", "mrr"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--caption", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = load(a.formats_dir, a.baseline_dir)

    fig, panels = plt.subplots(1, 2, figsize=(11.6, 5.4), sharey=True)
    xs = range(len(FORMS))
    rows = []
    for ax, (direction, title) in zip(panels, DIRECTIONS):
        for arch, size in MODELS:
            ys, errs = [], []
            for form, _ in FORMS:
                v = data.get(((arch, size), form, direction))
                if not v:
                    ys.append(float("nan"))
                    errs.append(0.0)
                    continue
                p, n = v[a.metric], v["queries"]
                ys.append(p)
                share = v["recall@1"]
                errs.append(1.96 * math.sqrt(max(share, 1e-4) * (1 - share) / n)
                            if a.metric == "recall@1" else 0.0)
                rows.append((arch, size, form, direction, p, v["queries"]))
            ax.errorbar(list(xs), ys, yerr=errs if a.metric == "recall@1" else None,
                        fmt="o", ms=5, capsize=3, linewidth=1.4, elinewidth=1.0,
                        linestyle=STYLE[arch], color=SIZE_COLOUR[size],
                        label=f"{arch} {size}")
        ax.axhline(CHANCE[a.metric], color="#888888", linestyle=":", linewidth=1.2)
        # In the empty strip under the line: at the middle of the axis the lines
        # themselves cross this height.
        ax.annotate(f"偶然に選んだ場合 {CHANCE[a.metric]}", (0.98, 0.02),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8, color="#666666")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([label for _, label in FORMS], fontsize=9)
        ax.set_xlim(-0.35, len(FORMS) - 0.65)
        axes(ax, ("Recall@1" if a.metric == "recall@1" else "MRR")
             if direction == "molecule_to_text" else "", "入力の形")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)
    panels[0].legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")

    # Checked against the drawn numbers: all fourteen inst/smiles measurements sit above
    # chance, while every inst/selfies one is within noise of it. The split by whether
    # the corpus holds the pair (2026-10-27 judgment section 3) says who makes that:
    # on the 895 pairs it does not hold, GPT-2's eight measurements fall back to chance
    # and BERT's molecule->text three do not. Drawn here is all 1,000, so the title has
    # to carry that or the figure overstates what it shows.
    title = ("コーパスと同じ表記とマーカーで訊くと 14 測定すべてが偶然を上回る"
             " —— ただし GPT-2 の分は重なり 105 対が作っている")
    fig.suptitle(title, fontsize=11.5, color=INK, x=0.008, ha="left")
    caption = open(a.caption, encoding="utf-8").read().strip()
    line_h = (7.5 * 1.35) / (fig.get_size_inches()[1] * 72)
    lines = [ln for raw in caption.split("\n")
             for ln in textwrap.wrap(raw, int(fig.get_size_inches()[0] * 10))]
    fig.text(0.008, 0.012, "\n".join(lines), fontsize=7.5, color="#666666", va="bottom")
    fig.tight_layout(rect=(0.008, 0.012 + line_h * len(lines) + 0.03, 0.995, 0.93))
    name = f"molnl-retrieval-formats-{a.metric.replace('@', '')}"
    fig.savefig(os.path.join(a.out_dir, name + ".png"), dpi=150)
    with open(os.path.join(a.out_dir, name + ".tsv"), "w") as fh:
        fh.write("arch\tsize\tformat\tdirection\t" + a.metric + "\tqueries\n")
        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"  wrote {name}.png and .tsv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
