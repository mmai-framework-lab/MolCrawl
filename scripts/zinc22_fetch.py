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

The server cuts connections partway through under load -- a plain SSL EOF, not a
status code -- and it does so more often the larger the file. A transfer therefore
keeps its partial file between attempts and asks for the rest with a Range header, so
a drop costs the bytes still missing rather than the bytes already taken. Backoff is
measured in tens of seconds rather than ones, because retrying immediately is what
the server is objecting to.

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
BACKOFF = (5, 20, 60, 150, 300)


def _request(url, auth, offset):
    """One GET, asking only for the bytes after `offset` when there are any."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if auth:
        req.add_header("Authorization", "Basic " + b64encode(auth.encode()).decode())
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    return req


def fetch_one(row, base, dest, auth, timeout, retries):
    """Download one file, resuming its partial across attempts. Never raises."""
    tranche, name = row["tranche"], row["file"]
    listed = int(row["size_bytes"])
    out_dir = os.path.join(dest, tranche)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    tmp = path + ".part"

    if os.path.exists(path):
        have = os.path.getsize(path)
        if listed == 0 or abs(have - listed) <= listed * SIZE_TOLERANCE:
            return {"tranche": tranche, "file": name, "status": "skipped",
                    "bytes": have, "seconds": 0.0, "attempts": 0, "error": ""}

    url = f"{base.rstrip('/')}/{tranche}/{name}"
    last = ""
    t0 = time.time()
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])
        offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        try:
            with urllib.request.urlopen(_request(url, auth, offset), timeout=timeout) as resp:
                # A server that ignores Range answers 200 with the whole file, so the
                # partial has to be thrown away rather than appended to.
                if offset and resp.status != 206:
                    offset = 0
                mode = "ab" if offset else "wb"
                with open(tmp, mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
            got = os.path.getsize(tmp)
            # A cut connection leaves a short file and no error, so the listed size is
            # the only check available before the gzip is read.
            if listed and got < listed * (1 - SIZE_TOLERANCE):
                last = f"short: {got} of about {listed}"
                continue
            os.replace(tmp, path)
            return {"tranche": tranche, "file": name, "status": "taken",
                    "bytes": got, "seconds": round(time.time() - t0, 1),
                    "attempts": attempt + 1, "error": ""}
        except urllib.error.HTTPError as exc:
            # 416 means the partial already runs to the end of the file. The listed
            # size is a rounded figure, so it can call a complete transfer short;
            # the server disagreeing is the better authority.
            if exc.code == 416 and offset:
                os.replace(tmp, path)
                return {"tranche": tranche, "file": name, "status": "taken",
                        "bytes": offset, "seconds": round(time.time() - t0, 1),
                        "attempts": attempt + 1, "error": "accepted on 416"}
            last = f"HTTPError: {exc.code} {exc.reason}"
        except Exception as exc:                    # noqa: BLE001 - recorded, not raised
            last = f"{type(exc).__name__}: {exc}"
            # The partial is deliberately kept: the next attempt resumes from it.
    return {"tranche": tranche, "file": name, "status": "failed",
            "bytes": os.path.getsize(tmp) if os.path.exists(tmp) else 0,
            "seconds": round(time.time() - t0, 1),
            "attempts": retries + 1, "error": last}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--base-url", default="https://files.docking.org/zinc22/2d")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--retry-list", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=5)
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
        w = csv.DictWriter(rec, fieldnames=["tranche", "file", "status", "bytes",
                                            "seconds", "attempts", "error"],
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
