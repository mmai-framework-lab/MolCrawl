"""rna の損失曲線を 1 行 1 評価点の TSV にする。

GPT-2 と BERT を 1 つの表に入れる。両者の損失は比べられないが、**同じ表に
入れることと比べてよいことは別である**。並べる側が metric 列を見て選べるよう、
実際に測った量の名前をそのまま持たせ、どちらの枠組みのどの量かが行から分かる
ようにしてある。

訓練側の値も入れる。GPT-2 は logging_*.csv が train と val を同じ行に持ち、
BERT は trainer_state.json の log_history に ``loss`` として現れる。

eval_set 列は「固定か引き直しか」を書く。三通りある:

``fixed-3200``
    GPT-2 で ``deterministic_val_eval`` が立っている run。valid の先頭から
    seed 12345 で並べ替えた 3,200 系列を毎回読む。
``resampled``
    GPT-2 でそれが無い run。評価のたびに検証行を引き直すので、散らばりが
    0.010-0.012 ある。
``rows-fixed/mask-redrawn``
    BERT。評価行は valid の先頭取りで固定だが、**マスクは評価のたびに引き直す**。
    rna の config は ``fixed_eval_mask`` を宣言していない。同じモデルを二度測ると
    引き方の分だけ違う。

job 列は、その step を走らせたジョブである。GPT-2 は区間ごとに
logging_*.csv が分かれるので、ファイルの順と job id の昇順を対応させる
（link は afterany で繋いであり、id の順に走る）。BERT は segments.log が
区間と job id を直に持つので、checkpoint の書き込み時刻がどの区間の窓に
入るかで決める。
"""
import argparse
import csv
import glob
import json
import os
import re

GPT2_RE = re.compile(r"rna-gpt2-(small|medium|large|xl)-lr([0-9a-z]+?)(?:-seed(\d+))?(?:-(9ep))?$")
BERT_RE = re.compile(r"rna-(small|medium|large|xl)-lr([0-9a-z]+)$")
LR_NUM = {"6e4": 6e-4, "3e4": 3e-4, "1p5e4": 1.5e-4, "1p2e3": 1.2e-3, "1e4": 1e-4,
          "1e3": 1e-3, "7p5e5": 7.5e-5, "5e5": 5e-5, "2p5e5": 2.5e-5, "1p2e5": 1.2e-5,
          "6e6": 6e-6, "1p3e4": 1.3e-4, "1p7e4": 1.7e-4, "2p3e4": 2.3e-4,
          "4p1e4": 4.1e-4, "5p5e4": 5.5e-4, "7p4e4": 7.4e-4, "1p4e4": 1.4e-4, "2e4": 2e-4}
COLUMNS = ["run", "arch", "size", "lr", "epochs", "seed", "step", "split",
           "metric", "value", "eval_set", "job"]
# 1 epoch = 訓練 34,407,040 系列 / グローバルバッチ 2,560。両経路で同じ。
STEPS_PER_EPOCH = 34407040 / 2560


def epochs_of(max_steps):
    """計画されたエポック数。3 と 9 のどちらかに丸める。"""
    e = max_steps / STEPS_PER_EPOCH
    return 9 if e > 6 else 3


def job_ids_for(run, log_dirs):
    """run の slurm ログから job id を昇順で返す。link は id の順に走る。"""
    ids = set()
    for d in log_dirs:
        for f in glob.glob(f"{d}/*{run}-*.out") + glob.glob(f"{d}/*{run}*.out"):
            m = re.search(r"-(\d+)\.out$", os.path.basename(f))
            if m and run in os.path.basename(f):
                ids.add(int(m.group(1)))
    return sorted(ids)


def read_gpt2(d, log_dirs):
    """(step, train, val, job) を区間の順に。同じ step は後の区間を採る。"""
    files = sorted(glob.glob(f"{d}/logging_*.csv"))
    jobs = job_ids_for(os.path.basename(d), log_dirs)
    seen = {}
    for seg, f in enumerate(files):
        job = jobs[seg] if seg < len(jobs) else (jobs[-1] if jobs else "")
        with open(f, newline="") as fh:
            for r in csv.reader(fh):
                if len(r) < 3 or not r[0].strip().isdigit():
                    continue
                seen[int(r[0])] = (float(r[1]), float(r[2]), job)
    return sorted((s,) + v for s, v in seen.items())


def bert_segment_jobs(d):
    """(開始時刻, job) を segments.log から。無ければ空。"""
    p = os.path.join(d, "segments.log")
    out = []
    if not os.path.exists(p):
        return out
    import datetime
    for line in open(p):
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            t = datetime.datetime.fromisoformat(parts[2]).timestamp()
        except ValueError:
            continue
        out.append((t, parts[1]))
    return sorted(out)


def bert_step_job(d):
    """step -> job。checkpoint の書き込み時刻が、どの区間の窓に入るかで決める。"""
    segs = bert_segment_jobs(d)
    if not segs:
        return {}
    marks = []
    for p in glob.glob(f"{d}/checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", p)
        if m:
            marks.append((int(m.group(1)), os.path.getmtime(p)))
    marks.sort()
    out = {}
    for step, mt in marks:
        job = segs[0][1]
        for t, j in segs:
            if t <= mt:
                job = j
        out[step] = job
    return out


def read_bert(d):
    """(step, metric, value) を trainer_state.json から重複なく。"""
    rows = {}
    maxstep = 0
    for st in glob.glob(f"{d}/checkpoint-*/trainer_state.json"):
        try:
            j = json.load(open(st))
        except (OSError, ValueError):
            continue
        maxstep = max(maxstep, int(j.get("max_steps") or 0))
        for h in j.get("log_history", []):
            s = h.get("step")
            if s is None:
                continue
            for k, name, split in (("loss", "train_loss", "train"),
                                   ("eval_loss", "eval_loss", "val"),
                                   ("eval_loss_mask", "eval_loss_mask", "val"),
                                   ("eval_loss_copy", "eval_loss_copy", "val"),
                                   ("eval_loss_random", "eval_loss_random", "val")):
                if k in h:
                    rows[(int(s), name, split)] = float(h[k])
    return rows, maxstep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpt2-root", required=True)
    ap.add_argument("--bert-root", required=True)
    ap.add_argument("--log-dir", action="append", default=[],
                    help="slurm ログの置き場。job 列のため。複数可")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    n = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(COLUMNS)

        for d in sorted(glob.glob(f"{a.gpt2_root}/rna-gpt2-*")):
            base = os.path.basename(d)
            m = GPT2_RE.match(base)
            if not m:
                continue
            mf = {}
            try:
                mf = json.load(open(os.path.join(d, "run_manifest.json")))
            except (OSError, ValueError):
                pass
            seed = (mf.get("seed") or {}).get("seed", m.group(3) or 42)
            det = (mf.get("eval") or {}).get("deterministic_val_eval")
            eval_set = "fixed-3200" if det else "resampled"
            maxit = (mf.get("schedule") or {}).get("max_iters") or (120960 if m.group(4) else 40320)
            ep = epochs_of(maxit)
            lr = (mf.get("schedule") or {}).get("learning_rate") or LR_NUM.get(m.group(2), m.group(2))
            for step, tr, va, job in read_gpt2(d, a.log_dir):
                for split, name, val in (("train", "train_loss", tr), ("val", "val_loss", va)):
                    w.writerow([base, "gpt2", m.group(1), lr, ep, seed, step,
                                split, name, f"{val:.6f}", eval_set, job])
                    n += 2

        for d in sorted(glob.glob(f"{a.bert_root}/rna-*")):
            base = os.path.basename(d)
            m = BERT_RE.match(base)
            if not m:
                continue
            rows, maxstep = read_bert(d)
            if not rows:
                continue
            ep = epochs_of(maxstep or 120960)
            sj = bert_step_job(d)
            jobs_sorted = sorted(sj.items())
            lr = LR_NUM.get(m.group(2), m.group(2))
            for (step, name, split), val in sorted(rows.items()):
                job = sj.get(step, "")
                if not job and jobs_sorted:
                    job = min(jobs_sorted, key=lambda kv: abs(kv[0] - step))[1]
                w.writerow([base, "bert", m.group(1), lr, ep, 42, step, split,
                            name, f"{val:.6f}", "rows-fixed/mask-redrawn", job])
                n += 1
    print(f"  {n:,} 行を {a.out} に書いた")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
