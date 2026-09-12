"""wheel を落として中身を読むだけ。導入はしない。

上長 §2.2 の「bras / sbee / CiLISI を触って確認してほしい」に、環境を汚さずに答える。
あわせて、どのパッケージが依存解決を壊すかを 1 つずつ切り分ける。
"""
import subprocess
import sys
import zipfile
import glob
import os
import re
def LOG(*a): print(*a, flush=True)
PY_ = sys.executable
ap = __import__("argparse").ArgumentParser(description=__doc__)
ap.add_argument("--work-dir", required=True, help="wheel の置き場（作業用）")
ap.add_argument("--packages", default="scib-metrics,jax,harmonypy,scvi-tools,scikit-misc",
                help="1 つずつ dry-run して依存解決を切り分ける対象")
A = ap.parse_args()
DEST = f"{A.work_dir}/wheels"
os.makedirs(DEST, exist_ok=True)

def run(cmd, t=900):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
    return p.returncode, (p.stdout or "") + (p.stderr or "")

LOG("=== 1. scib-metrics の中身（wheel を読むだけ）===")
rc, out = run([PY_, "-m", "pip", "download", "--no-deps", "-d", DEST, "scib-metrics"])
LOG(f"  download rc={rc}")
whl = sorted(glob.glob(f"{DEST}/scib_metrics-*.whl"))
if whl:
    LOG(f"  wheel: {os.path.basename(whl[-1])}")
    with zipfile.ZipFile(whl[-1]) as z:
        names = z.namelist()
        init = [n for n in names if n.endswith("scib_metrics/__init__.py")]
        if init:
            src = z.read(init[0]).decode()
            LOG("  --- __init__.py の公開名 ---")
            m = re.search(r"__all__\s*=\s*\[(.*?)\]", src, re.S)
            pub = re.findall(r"[\"']([^\"']+)[\"']", m.group(1)) if m else \
                  re.findall(r"^from .+ import (.+)$", src, re.M)
            LOG("   " + ", ".join(sorted(set(x.strip() for x in pub))))
        LOG("  --- モジュール一覧 ---")
        mods = sorted(n for n in names if n.endswith(".py") and "/_" not in n.rsplit("/",1)[0])
        LOG("   " + ", ".join(os.path.basename(n) for n in mods)[:900])
        LOG("  --- 指定の 3 指標を全ファイルから検索 ---")
        blob = "\n".join(z.read(n).decode("utf-8", "ignore") for n in names if n.endswith(".py"))
        for key in ["bras", "BRAS", "sbee", "SBEE", "cilisi", "CiLISI", "ilisi_knn",
                    "clisi_knn", "kbet", "silhouette_label", "silhouette_batch",
                    "nmi_ari_cluster_labels_kmeans", "isolated_labels", "graph_connectivity",
                    "pcr_comparison", "kbet_per_label"]:
            hits = len(re.findall(rf"\b{re.escape(key)}\b", blob))
            LOG(f"    {key:32s} {'あり' if hits else '見当たらない':10s} ({hits} 箇所)")
        LOG("  --- BioConservation / BatchCorrection の項目 ---")
        bm = [n for n in names if n.endswith("benchmark/_core.py") or n.endswith("_core.py")]
        for n in bm:
            src = z.read(n).decode("utf-8", "ignore")
            for cls in ["BioConservation", "BatchCorrection"]:
                mm = re.search(rf"class {cls}[^\n]*:\n(.*?)\n\n\n", src, re.S)
                if mm:
                    fields = re.findall(r"^\s{4}(\w+):\s*bool", mm.group(1), re.M)
                    LOG(f"    {cls}: {', '.join(fields)}")
    LOG("  --- 依存 ---")
    with zipfile.ZipFile(whl[-1]) as z:
        md = [n for n in z.namelist() if n.endswith("METADATA")]
        if md:
            for line in z.read(md[0]).decode().splitlines():
                if line.startswith("Requires-Dist"):
                    LOG("    " + line.replace("Requires-Dist: ", ""))

LOG("\n=== 2. どのパッケージが依存解決を壊すか（1 つずつ dry-run）===")
LOG("  注意: pip のキャッシュに wheel が残っていると、ソースビルドが要るはずの")
LOG("        パッケージも「解決できる」と出る。初回の答えを知りたい場合は")
LOG("        PIP_NO_CACHE_DIR=1 を付けて走らせること。")
for pkg in A.packages.split(","):
    rc, out = run([PY_, "-m", "pip", "install", "--dry-run", "--no-input", pkg])
    if rc == 0:
        n = len(re.findall(r"Would install", out)) and re.search(r"Would install (.+)", out)
        LOG(f"  {pkg:14s} ✅ 解決できる")
        if n:
            LOG(f"                 追加/更新: {n.group(1)[:400]}")
    else:
        bad = re.findall(r"(?:Preparing metadata|Getting requirements).*?\n.*?×.*?\n", out)
        why = re.search(r"× (.+)", out)
        LOG(f"  {pkg:14s} ❌ rc={rc}  {why.group(1) if why else out.strip().splitlines()[-1][:120]}")
        m2 = re.search(r"pip-install-\w+/([a-zA-Z0-9_.-]+)_", out)
        if m2:
            LOG(f"                 壊しているのは: {m2.group(1)}")
