# -*- coding: utf-8 -*-
"""
TROPOMI 大气成分下载脚本 — 质量优先版

用途: 大气成源（源11）——火山(SO2)、沙尘/雾霾(气溶胶指数)训练数据
产品: S5P L2（1-Orbit，每天约14圈）
  - SO2____HiR   → 火山 SO2
  - AER_AI_HiR   → 气溶胶指数（沙尘/雾霾）
数据源: NASA GES DISC（Earthdata Bearer token）

⚠️ 单 orbit 文件 ~100-500MB，一天约 1.4-7GB，按需下（火山/沙尘事件时段）
质量校验: NetCDF4 文件头魔数 \\x89HDF + 大小下限

用法:
  python scripts/download_tropomi.py --product SO2 --start 2026-08-10
  python scripts/download_tropomi.py --product AER_AI --start 2026-08-10 --end 2026-08-11
"""
import os
from config_local import cred as _cred
import sys
import time
import json
import argparse
import subprocess
from datetime import date, timedelta

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOKEN = os.environ.get("EARTHDATA_TOKEN", "") or _cred("EARTHDATA_TOKEN")

PRODUCTS = {
    "SO2": "S5P_L2__SO2____HiR",
    "AER_AI": "S5P_L2__AER_AI_HiR",
}
OUT_DIR = r"C:\vlmdisaster\data\raw\11_atmospheric\tropomi"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"

NC_MAGIC = b"\x89HDF\r\n\x1a\n"
MIN_SIZE = 10_000_000  # 10MB，单 orbit 约 100-500MB


def query_cmr(short, day):
    """CMR 查询某天的 granule，返回 [(filename, url)]"""
    url = (f"{CMR}?short_name={short}"
           f"&temporal={day}T00:00:00Z,{day + timedelta(days=1)}T00:00:00Z"
           "&page_size=200")
    r = subprocess.run(["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke",
                        "--max-time", "90", "-s", url], capture_output=True)
    try:
        data = json.loads(r.stdout.decode("utf-8", errors="ignore"))
    except Exception:
        return []
    results = []
    for e in data.get("feed", {}).get("entry", []):
        title = e.get("title", "")
        fn = title.split(":", 1)[-1] if ":" in title else title
        for link in e.get("links", []):
            h = link.get("href", "")
            if h.endswith(".nc") and "gesdisc" in h:
                results.append((fn, h))
                break
    return results


def download_file(url, save_path, max_attempts=3):
    if os.path.exists(save_path):
        if os.path.getsize(save_path) >= MIN_SIZE:
            with open(save_path, "rb") as f:
                if f.read(8) == NC_MAGIC:
                    return "exists"
        os.remove(save_path)

    tmp = save_path + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass

    for attempt in range(1, max_attempts + 1):
        cmd = ["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke",
               "-H", f"Authorization: Bearer {TOKEN}",
               "--connect-timeout", "20", "--max-time", "1800",
               "-C", "-", "-s", "-o", tmp, url]
        try:
            subprocess.run(cmd, timeout=1860)
        except Exception:
            pass
        if os.path.exists(tmp) and os.path.getsize(tmp) >= MIN_SIZE:
            with open(tmp, "rb") as f:
                magic = f.read(8)
            if magic == NC_MAGIC:
                os.replace(tmp, save_path)
                return "ok"
            else:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return "bad"
        time.sleep(8)
    return "fail"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=str, required=True, choices=list(PRODUCTS), help="SO2 或 AER_AI")
    parser.add_argument("--start", type=str, required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期（默认=起始日）")
    args = parser.parse_args()

    short = PRODUCTS[args.product]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else start

    sub_dir = os.path.join(OUT_DIR, args.product.lower())
    os.makedirs(sub_dir, exist_ok=True)

    print("=" * 60)
    print(f"  TROPOMI {args.product} 下载")
    print("=" * 60)
    print(f"  范围: {start} ~ {end}")
    print(f"  输出: {sub_dir}")
    print(f"  ⚠️ 单 orbit ~100-500MB，一天约14圈，按需下")
    print()

    stats = {"ok": 0, "exists": 0, "bad": 0, "fail": 0}
    d = start
    while d <= end:
        dstr = d.isoformat()
        granules = query_cmr(short, d)
        if not granules:
            print(f"  [{dstr}] 无 granule（可能未发布）", flush=True)
        for fn, url in granules:
            save = os.path.join(sub_dir, fn)
            r = download_file(url, save)
            stats[r] += 1
            if r == "ok":
                print(f"  [{dstr}] ✓ {fn[:45]}... ({os.path.getsize(save)/1e6:.0f}MB)", flush=True)
            elif r == "exists":
                print(f"  [{dstr}] · 已有 {fn[:40]}", flush=True)
            elif r == "bad":
                print(f"  [{dstr}] ✗ 假文件 {fn[:40]}", flush=True)
            else:
                print(f"  [{dstr}] ✗ 失败 {fn[:40]}", flush=True)
            time.sleep(2)
        d += timedelta(days=1)

    print()
    print("=" * 60)
    print(f"  完成: 下载{stats['ok']} 已有{stats['exists']} 假文件{stats['bad']} 失败{stats['fail']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
