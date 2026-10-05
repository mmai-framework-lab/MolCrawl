"""List every ZINC-22 2D tranche file with its size, without downloading any of it.

The acquisition has to be able to say afterwards which tranches were taken and
which were missed, and that is only possible against a list fixed beforehand. The
server's own directory index carries the size of every file, so the total can be
measured from the listing alone -- the alternative, a HEAD per file, is thousands
of requests for the same answer.

Sizes come from Apache's human-readable column (1.0G, 175M, 881K), so they are
rounded to two significant figures: a total built from them is good to a few per
cent, not exact. The exact bytes arrive with the files.

ZINC-22's 2D tree is indexed by heavy-atom count (H04 .. H49) and, within each,
by logP -- M for negative, P for positive, the digits being logP x 100. Both are
kept as columns so a gap can be named by the region it falls in.

The tree is behind HTTP basic auth. Credentials come from the environment, never
from this file.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request
from base64 import b64encode

ROW = re.compile(
    r'<a href="(?P<name>[^"?/][^"]*)">[^<]*</a>\s*</td>'
    r'\s*<td align="right">(?P<mtime>[^<]*?)\s*</td>'
    r'\s*<td align="right">\s*(?P<size>[^<]*?)\s*</td>'
)
DIR = re.compile(r'<a href="(?P<name>[A-Za-z0-9_.-]+)/"')
UNITS = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def to_bytes(text: str) -> int | None:
    """Apache's rounded size column as bytes. '-' (a directory) returns None."""
    text = text.strip()
    if not text or text == "-":
        return None
    if text[-1] in UNITS:
        return int(float(text[:-1]) * UNITS[text[-1]])
    try:
        return int(text)
    except ValueError:
        return None


# The server runs mod_qos and answers urllib's default User-Agent with 403 where it
# answers curl's with 200, so the agent is set explicitly. It names the project and a
# contact rather than impersonating a browser: a server operator who wants to throttle
# or block this should be able to tell who it is.
USER_AGENT = "MolCrawl-dataprep/1.0 (research corpus build; +https://files.docking.org/zinc22)"


def fetch(url: str, auth: str | None, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if auth:
        req.add_header("Authorization", "Basic " + b64encode(auth.encode()).decode())
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode("utf-8", "replace")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True, help="the 2d/ directory")
    ap.add_argument("--out", required=True, help="TSV to write")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--auth-env", default="ZINC22_AUTH",
                    help="environment variable holding user:password")
    args = ap.parse_args(argv)

    auth = os.environ.get(args.auth_env) or None
    base = args.base_url.rstrip("/") + "/"
    tranches = sorted(set(DIR.findall(fetch(base, auth, args.timeout))))
    if not tranches:
        print(f"no tranches under {base}", file=sys.stderr)
        return 1
    print(f"tranches: {len(tranches)}  ({tranches[0]} .. {tranches[-1]})")

    rows, missing = [], []
    for t in tranches:
        try:
            page = fetch(f"{base}{t}/", auth, args.timeout)
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            missing.append((t, str(exc)))
            print(f"  {t:<6} 取得できない: {exc}")
            continue
        here = []
        for m in ROW.finditer(page):
            size = to_bytes(m.group("size"))
            if size is None:            # a directory row
                continue
            name = m.group("name")
            # H24M000.smi.gz -> heavy atoms 24, logP -0.00
            lp = re.match(r"H(\d+)([MP])(\d+)", name)
            here.append({
                "tranche": t, "file": name,
                "heavy_atoms": lp.group(1) if lp else "",
                "logp": (("-" if lp.group(2) == "M" else "+") + lp.group(3)) if lp else "",
                "size_text": m.group("size"), "size_bytes": size,
                "last_modified": m.group("mtime").strip(),
            })
        rows.extend(here)
        print(f"  {t:<6} {len(here):>5} ファイル  {sum(r['size_bytes'] for r in here) / 1024**3:>9.1f} GiB")

    cols = ["tranche", "file", "heavy_atoms", "logp", "size_text", "size_bytes", "last_modified"]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    total = sum(r["size_bytes"] for r in rows)
    print(f"\nwrote {args.out}")
    print(f"ファイル総数 {len(rows):,}")
    print(f"圧縮後の総バイト {total:,} = {total / 1024**4:.2f} TiB")
    if missing:
        print(f"\n取得できなかった区画 {len(missing)}:")
        for t, why in missing:
            print(f"  {t}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
