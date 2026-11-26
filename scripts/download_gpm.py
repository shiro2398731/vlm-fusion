# -*- coding: utf-8 -*-
"""
GPM IMERG 降水下载脚本 — 质量优先版（v2，Late Run）

用途: 雷达降水源（源6）——飓风/洪水降水结构识别训练数据
产品: GPM_3IMERGHHL（IMERG Late Run 半小时，滞后约12小时，近实时可用）
数据源: NASA GES DISC（Earthdata Bearer token）

⚠️ 说明:
  - Late Run 半小时产品，一天 48 个文件（每半小时一个）
  - 单文件 ~10-20MB，一天约 500MB-1GB，按需下（飓风/洪水事件时段）
  - 处理阶段再把 48 个半小时累计成日降水

质量校验: HDF5 文件头魔数 \\x89HDF + 大小下限，防登录页/错误页假文件

用法:
  python scripts/download_gpm.py --start 2026-08-10
  python scripts/download_gpm.py --start 2026-08-10 --end 2026-08-12
"""
import os
from config_local import cred as _cred
import sys
import time
import argparse
import subprocess
from datetime import date, timedelta

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOKEN = os.environ.get("EARTHDATA_TOKEN", "") or _cred("EARTHDATA_TOKEN")

PRODUCT = "GPM_3IMERGHHL"
OUT_DIR = r"C:\vlmdisaster\data\raw\06_radar_precipitation\gpm"
BASE = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3"

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
MIN_SIZE = 1_000_000  # 1MB，半小时文件约 10-20MB，防假文件


def halfhour_names(day):
    """构造某天的 48 个半小时 IMERG 文件名"""
    ymd = day.strftime("%Y%m%d")
    names = []
    for i in range(48):
        total_min = i * 30
        sh, sm = divmod(total_min, 60)
        start = f"S{sh:02d}{sm:02d}00"
        end_total = total_min + 29
        eh, em = divmod(end_total, 60)
        end = f"E{eh:02d}{em:02d}59"
        fn = f"3B-HHR-L.MS.MRG.3IMERG.{ymd}-{start}-{end}.{i*30:04d}.V07C.HDF5"
        names.append(fn)
    return names


def download_file(url, save_path, max_attempts=3):
    if os.path.exists(save_path):
        if os.path.getsize(save_path) >= MIN_SIZE:
            with open(save_path, "rb") as f:
                if f.read(8) == HDF5_MAGIC:
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
                         magic=HDF5_MAGIC, resume=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期（默认=起始日）")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else start

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  GPM IMERG (Late Run 半小时) 下载")
    print("=" * 60)
    print(f"  范围: {start} ~ {end}（每天 48 个半小时文件）")
    print(f"  输出: {OUT_DIR}")
    print()

    stats = {"ok": 0, "exists": 0, "bad": 0, "fail": 0}
    d = start
    while d <= end:
        doy = d.timetuple().tm_yday
        day_ok = 0
        for fn in halfhour_names(d):
            url = f"{BASE}/{PRODUCT}.07/{d.year}/{doy:03d}/{fn}"
            save = os.path.join(OUT_DIR, fn)
            r = download_file(url, save)
            stats[r] += 1
            if r in ("ok", "exists"):
                day_ok += 1
            elif r == "bad":
                print(f"  [{d} {fn[-26:-5]}] ✗ 假文件", flush=True)
            else:
                print(f"  [{d} {fn[-26:-5]}] ✗ 失败", flush=True)
            time.sleep(1)  # 限流
        print(f"  [{d}] 当日已下 {day_ok}/48 个文件", flush=True)
        d += timedelta(days=1)

    print()
    print("=" * 60)
    print(f"  完成: 下载{stats['ok']} 已有{stats['exists']} 假文件{stats['bad']} 失败{stats['fail']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
