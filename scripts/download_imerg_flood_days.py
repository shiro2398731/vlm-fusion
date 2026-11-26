# -*- coding: utf-8 -*-
r"""按 GDACS 洪水事件下载 IMERG 日产品（3B-DAY-L）前后各 2 天。

用法：
  .\venv\Scripts\python.exe scripts\download_imerg_flood_days.py --top 15
"""
import argparse
from config_local import cred as _cred
import datetime
import json
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EARTHDATA_TOKEN = os.environ.get("EARTHDATA_TOKEN", "") or _cred("EARTHDATA_TOKEN")

ROOT = r"C:\vlmdisaster"
OUT = os.path.join(ROOT, "data", "raw", "06_radar_precipitation", "imerg_day")
GDACS = os.path.join(ROOT, "data", "raw", "14_gdacs_floods.json")
BASE = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGDL.07"


def events(top):
    d = json.load(open(GDACS, encoding="utf-8"))
    ev = d if isinstance(d, list) else d.get("features", d.get("events", []))
    out = []
    for e in ev:
        p = e.get("properties", e)
        dt = (p.get("fromdate") or "")[:10]
        if not dt:
            continue
        out.append({"date": dt, "country": p.get("country", ""), "alert": p.get("alertlevel", "")})
    out.sort(key=lambda x: x["date"], reverse=True)
    return out[:top]


def path_for(d):
    fn = "3B-DAY-L.MS.MRG.3IMERG.%s-S000000-E235959.V07B.nc4" % d.strftime("%Y%m%d")
    return "%s/%d/%02d/%s" % (BASE, d.year, d.month, fn), fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--span", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ev = events(a.top)
    days = set()
    for e in ev:
        d0 = datetime.date.fromisoformat(e["date"])
        for k in range(-a.span, a.span + 1):
            days.add(d0 + datetime.timedelta(days=k))
    days = sorted(days)
    print("事件 %d，日期 %d 天" % (len(ev), len(days)))

    ok = fail = skip = 0
    for i, d in enumerate(days, 1):
        url, fn = path_for(d)
        dst = os.path.join(OUT, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 5_000_000:
            skip += 1
            continue
        part = dst + ".part"
        subprocess.run(["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke", "--max-time", "600",
                        "-s", "-H", "Authorization: Bearer " + EARTHDATA_TOKEN, "-o", part, url],
                       capture_output=True, text=True)
        good = False
        if os.path.exists(part) and os.path.getsize(part) > 5_000_000:
            with open(part, "rb") as f:
                good = f.read(8).startswith(b"\x89HDF") or f.read(4) == b"CDF"
            if good:
                os.replace(part, dst)
                ok += 1
        if not good:
            fail += 1
            if os.path.exists(part):
                os.remove(part)
        if i % 10 == 0 or i == len(days):
            print("  [%d/%d] ok=%d skip=%d fail=%d" % (i, len(days), ok, skip, fail), flush=True)
    print("完成: ok=%d skip=%d fail=%d -> %s" % (ok, skip, fail, OUT))


if __name__ == "__main__":
    main()
