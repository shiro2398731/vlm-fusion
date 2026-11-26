# -*- coding: utf-8 -*-
"""
MODIS MOD08_D3（每日全球气溶胶 AOD）下载脚本 — 质量优先版

用途: 大气成源（源11）——雾/霾识别训练数据
产品: MOD08_D3（Terra 每日全球气溶胶/光学厚度，不分 tile，全球一个文件）
数据源: NASA LAADS archive（Earthdata Bearer token）

质量校验:
  1. HDF4 文件头魔数 \\x0e\\x03\\x13\\x01 —— 防登录页/错误页假文件
  2. 文件大小 >= 50MB —— MOD08_D3 全球文件约 161MB，防截断

⚠️ 单文件约 161MB，按需下（雾霾事件时段），别一次下全年爆盘

用法:
  python scripts/download_mod08.py --start 2026-08-01 --end 2026-08-05
"""
import os
from config_local import cred as _cred
import re
import sys
import time
import argparse
import subprocess
from datetime import date, timedelta

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOKEN = os.environ.get("EARTHDATA_TOKEN", "") or _cred("EARTHDATA_TOKEN")

PRODUCT = "MOD08_D3"
OUT_DIR = r"C:\vlmdisaster\data\raw\11_atmospheric\mod08"
BASE = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61"

HDF4_MAGIC = b"\x0e\x03\x13\x01"
MIN_SIZE = 50_000_000  # 50MB，MOD08_D3 全球文件约 161MB，防假文件/截断


from dl_common import curl_get, http_download


def list_dir_file(d):
    """解析某日期的 LAADS 目录页，返回 MOD08_D3 文件名"""
    doy = d.timetuple().tm_yday
    url = f"{BASE}/{PRODUCT}/{d.year}/{doy:03d}/"
    html = curl_get(url, 60, [f"Authorization: Bearer {TOKEN}"])
    text = html.decode("utf-8", errors="ignore")
    for m in re.finditer(r'href="([^"]+\.hdf)"', text):
        full = m.group(1)
        fn = full.split("/")[-1]
        if fn.startswith("MOD08_D3"):
            return fn
    return None


def download_file(url, save_path, max_attempts=3):
    if os.path.exists(save_path):
        if os.path.getsize(save_path) >= MIN_SIZE:
            with open(save_path, "rb") as f:
                if f.read(4) == HDF4_MAGIC:
                    return "exists"
        os.remove(save_path)

    tmp = save_path + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass

    from dl_common import http_download
    return http_download(url, save_path,
                         headers=[f"Authorization: Bearer {TOKEN}"],
                         max_attempts=max_attempts, min_size=MIN_SIZE,
                         magic=HDF4_MAGIC, resume=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期（默认=起始日）")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else start

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  MODIS MOD08_D3（每日全球 AOD）下载")
    print("=" * 60)
    print(f"  范围: {start} ~ {end}")
    print(f"  输出: {OUT_DIR}")
    print(f"  ⚠️ 单文件约 161MB，按需下")
    print()

    stats = {"ok": 0, "exists": 0, "bad": 0, "fail": 0}
    d = start
    while d <= end:
        dstr = d.isoformat()
        fn = list_dir_file(d)
        if not fn:
            print(f"  [{dstr}] 目录无数据（可能未发布）", flush=True)
        else:
            url = f"{BASE}/{PRODUCT}/{d.year}/{d.timetuple().tm_yday:03d}/{fn}"
            save = os.path.join(OUT_DIR, fn)
            r = download_file(url, save)
            stats[r] += 1
            if r == "ok":
                print(f"  [{dstr}] ✓ 已下载 ({os.path.getsize(save)/1e6:.0f}MB)", flush=True)
            elif r == "exists":
                print(f"  [{dstr}] · 已有", flush=True)
            elif r == "bad":
                print(f"  [{dstr}] ✗ 假文件（非HDF）", flush=True)
            else:
                print(f"  [{dstr}] ✗ 失败", flush=True)
        d += timedelta(days=1)
        time.sleep(2)

    print()
    print("=" * 60)
    print(f"  完成: 下载{stats['ok']} 已有{stats['exists']} 假文件{stats['bad']} 失败{stats['fail']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
