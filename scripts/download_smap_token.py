# -*- coding: utf-8 -*-
r"""
SMAP L3 土壤湿度下载器（Bearer token 版）。

之前 download_smap.py 下载到的文件是假文件，原因是走了别的下载函数；
这里直接用 CMR 的 granule URL + Authorization: Bearer <token>。

用法：
  .\venv\Scripts\python.exe scripts\download_smap_token.py --start 2020-01-01 --end 2022-12-31 --step 9
"""
import argparse
from config_local import cred as _cred
import datetime
import glob
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EARTHDATA_TOKEN = os.environ.get("EARTHDATA_TOKEN", "") or _cred("EARTHDATA_TOKEN")

ROOT = r"C:\vlmdisaster"
OUT = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "smap2")
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
        url = ("%s?short_name=SPL3SMP&temporal=%sT00:00:00Z,%sT00:00:00Z"
               "&page_size=2000&page_num=%d" % (CMR, start, end, page))
        j = curl_json(url)
        items = j.get("items", [])
        if not items:
            break
        for it in items:
            umm = it["umm"]
            upd = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
            d = upd.get("BeginningDateTime", "")[:10]
            urls = [x.get("URL", "") for x in umm.get("RelatedUrls", [])]
            h5 = [u for u in urls if u.endswith(".h5")]
            if d and h5:
                out.setdefault(d, h5[0])
        if len(items) < 2000:
            break
        page += 1
    return out


def download(url, dst):
    part = dst + ".part"
    r = subprocess.run(["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke",
                        "--max-time", "900", "-s", "-H", "Authorization: Bearer " + EARTHDATA_TOKEN,
                        "-o", part, url], capture_output=True, text=True)
    if os.path.exists(part) and os.path.getsize(part) > 5_000_000:
        with open(part, "rb") as f:
            magic = f.read(8)
        if magic.startswith(b"\x89HDF"):
            os.replace(part, dst)
            return "ok"
    if os.path.exists(part):
        os.remove(part)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step", type=int, default=9, help="每 N 天取一天")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print("查询 CMR ...", flush=True)
    gr = list_granules(a.start, a.end)
    print("可用日期 %d 个" % len(gr), flush=True)
    days = sorted(gr)
    picked = []
    for i, d in enumerate(days):
        if i % a.step == 0:
            picked.append(d)
    print("按 step=%d 取 %d 天" % (a.step, len(picked)), flush=True)

    stats = {"ok": 0, "skip": 0, "fail": 0}
    for i, d in enumerate(picked, 1):
        dst = os.path.join(OUT, os.path.basename(gr[d]))
        if os.path.exists(dst) and os.path.getsize(dst) > 5_000_000:
            stats["skip"] += 1
            continue
        res = download(gr[d], dst)
        stats[res] += 1
        print("  [%d/%d] %s %s %.0fMB" % (i, len(picked), d, res,
                                           os.path.getsize(dst) / 1e6 if os.path.exists(dst) else 0), flush=True)
    print("完成:", stats, "->", OUT)


if __name__ == "__main__":
    main()
