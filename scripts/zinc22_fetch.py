"""Fetch ZINC-22 tranche files from a fixed list, recording what was and was not taken.

Works from the inventory written by zinc22_tranche_inventory.py rather than from a
live listing, so "which files should exist" is decided once and the acquisition can
be checked against it afterwards. A file already on disk at its listed size is
skipped, which makes the whole thing resumable: re-running after an interruption
costs one stat per file.

Every file gets a row in the record -- taken, skipped, or failed -- and the failures
are written to their own list so a second pass has something to work from. An empty
retry list is the only evidence that nothing was missed.

Sizes in the inventory are Apache's rounded column, so a downloaded file is accepted
when it is within a few per cent of the listed size rather than equal to it.

Credentials come from the environment, never from this file.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import urllib.error
import urllib.request
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed

USER_AGENT = "MolCrawl-dataprep/1.0 (research corpus build)"
SIZE_TOLERANCE = 0.12      # Apache rounds to two significant figures


def fetch_one(row, base, dest, auth, timeout, retries):
    """Download one file. Returns a record row; never raises."""
    tranche, name = row["tranche"], row["file"]
    listed = int(row["size_bytes"])
    out_dir = os.path.join(dest, tranche)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)

    if os.path.exists(path):
        have = os.path.getsize(path)
        if listed == 0 or abs(have - listed) <= listed * SIZE_TOLERANCE:
            return {"tranche": tranche, "file": name, "status": "skipped",
                    "bytes": have, "seconds": 0.0, "error": ""}

    url = f"{base.rstrip('/')}/{tranche}/{name}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if auth:
        req.add_header("Authorization", "Basic " + b64encode(auth.encode()).decode())

    last = ""
    for attempt in range(retries + 1):
        t0 = time.time()
        tmp = path + ".part"
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            got = os.path.getsize(tmp)
            # A truncated transfer leaves a short file and no error; the listed size
            # is the only check available before the gzip is read.
            if listed and got < listed * (1 - SIZE_TOLERANCE):
                raise OSError(f"short: {got} of about {listed}")
            os.replace(tmp, path)
            return {"tranche": tranche, "file": name, "status": "taken",
                    "bytes": got, "seconds": round(time.time() - t0, 1), "error": ""}
        except Exception as exc:                    # noqa: BLE001 - recorded, not raised
            last = f"{type(exc).__name__}: {exc}"
            if os.path.exists(tmp):
                os.unlink(tmp)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return {"tranche": tranche, "file": name, "status": "failed",
            "bytes": 0, "seconds": 0.0, "error": last}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--base-url", default="https://files.docking.org/zinc22/2d")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--retry-list", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--only-tranches", nargs="*", default=[],
                    help="restrict to these tranches, for splitting across jobs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    auth = os.environ.get("ZINC22_AUTH") or None
    rows = list(csv.DictReader(open(args.inventory, encoding="utf-8"), delimiter="\t"))
    if args.only_tranches:
        keep = set(args.only_tranches)
        rows = [r for r in rows if r["tranche"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    want = sum(int(r["size_bytes"]) for r in rows)
    print(f"対象 {len(rows):,} ファイル / 約 {want / 1024**4:.2f} TiB  並列 {args.workers}", flush=True)

    os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
    done = taken = skipped = failed = 0
    got_bytes = 0
    t0 = time.time()
    with open(args.record, "w", encoding="utf-8", newline="") as rec:
        w = csv.DictWriter(rec, fieldnames=["tranche", "file", "status", "bytes", "seconds", "error"],
                           delimiter="\t")
        w.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, r, args.base_url, args.dest, auth,
                                   args.timeout, args.retries) for r in rows]
            for fut in as_completed(futures):
                res = fut.result()
                w.writerow(res)
                rec.flush()
                done += 1
                got_bytes += res["bytes"]
                if res["status"] == "taken":
                    taken += 1
                elif res["status"] == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"  失敗 {res['tranche']}/{res['file']}: {res['error']}", flush=True)
                if done % 25 == 0 or done == len(rows):
                    el = time.time() - t0
                    print(f"  {done:,}/{len(rows):,}  取得 {taken:,} 既存 {skipped:,} 失敗 {failed:,}  "
                          f"{got_bytes / 1024**3:.1f} GiB  {got_bytes / el / 1048576:.1f} MiB/s  "
                          f"経過 {el / 60:.1f} 分", flush=True)

    with open(args.retry_list, "w", encoding="utf-8") as fh:
        for r in csv.DictReader(open(args.record, encoding="utf-8"), delimiter="\t"):
            if r["status"] == "failed":
                fh.write(f"{r['tranche']}\t{r['file']}\t{r['error']}\n")

    el = time.time() - t0
    print(f"\n完了 {done:,} ファイル  取得 {taken:,} / 既存 {skipped:,} / 失敗 {failed:,}")
    print(f"取得バイト {got_bytes:,} = {got_bytes / 1024**4:.2f} TiB  所要 {el / 3600:.2f} 時間")
    print(f"記録 {args.record}")
    print(f"再取得の一覧 {args.retry_list}" + ("（空）" if not failed else f"（{failed} 件）"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
