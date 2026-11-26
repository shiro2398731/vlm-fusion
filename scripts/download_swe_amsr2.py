# -*- coding: utf-8 -*-
r"""
AMSR2 日积雪当量（AU_DySno）下载器（Bearer token）。

用法：
  .\venv\Scripts\python.exe scripts\download_swe_amsr2.py --start 2020-11-01 --end 2021-03-31 --step 5
"""
import argparse
from config_local import cred as _cred
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EARTHDATA_TOKEN = os.environ.get("EARTHDATA_TOKEN", "") or _cred("EARTHDATA_TOKEN")

ROOT = r"C:\vlmdisaster"
OUT = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "swe_amsr2")
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"


def curl_json(url):
    r = subprocess.run(["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke",
                        "--max-time", "120", "-s", url], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def list_granules(start, end):
    out = {}
    page = 1
    while True:
        url = ("%s?short_name=AU_DySno&temporal=%sT00:00:00Z,%sT23:59:59Z"
               "&page_size=2000&page_num=%d" % (CMR, start, end, page))
        j = curl_json(url)
        items = j.get("items", [])
        if not items:
            break
        for it in items:
            umm = it["umm"]
            d = umm.get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime", "")[:10]
            urls = [x.get("URL", "") for x in umm.get("RelatedUrls", [])]
            he5 = [u for u in urls if u.endswith(".he5")]
            if d and he5:
                out.setdefault(d, he5[0])
        if len(items) < 2000:
            break
        page += 1
    return out


def download(url, dst):
    part = dst + ".part"
    r = subprocess.run(["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke",
                        "--max-time", "600", "-s", "-H", "Authorization: Bearer " + EARTHDATA_TOKEN,
                        "-o", part, url], capture_output=True, text=True)
    if os.path.exists(part) and os.path.getsize(part) > 1_000_000:
        with open(part, "rb") as f:
            ok = f.read(8).startswith(b"\x89HDF")
        if ok:
            for _ in range(10):
                try:
                    os.replace(part, dst)
                    return "ok"
                except PermissionError:
                    time.sleep(1)
    if os.path.exists(part):
        try:
            os.remove(part)
        except PermissionError:
            pass
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("查询 CMR ...", flush=True)
    gr = list_granules(a.start, a.end)
    print("可用日期 %d 个" % len(gr), flush=True)
    days = sorted(gr)[::a.step]
    print("按 step=%d 取 %d 天" % (a.step, len(days)), flush=True)

    stats = {"ok": 0, "skip": 0, "fail": 0}
    for i, d in enumerate(days, 1):
        dst = os.path.join(OUT, os.path.basename(gr[d]))
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            stats["skip"] += 1
            continue
        res = download(gr[d], dst)
        stats[res] += 1
        print("  [%d/%d] %s %s" % (i, len(days), d, res), flush=True)
    print("完成:", stats, "->", OUT)


if __name__ == "__main__":
    main()
