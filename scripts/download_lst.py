# -*- coding: utf-8 -*-
"""
MODIS 地表温度 (MOD11A2) 下载脚本 — 质量优先版

用途: 热红外-地表温度灾害（高温/寒潮/霜冻/冻害/干旱/城市热岛）训练数据
产品: MOD11A2 (Terra 8天合成, 1km 地表温度 LST)
数据源: NASA LAADS archive（需 Earthdata 认证，Bearer token）

质量校验（每文件必过，否则重下/丢弃）:
  1. HDF4 文件头魔数 \\x0e\\x03\\x13\\x01 —— 防止登录页/错误页假文件
  2. 文件大小 >= 1MB —— MOD11A2 单 tile 通常 2~5MB，防截断
  3. 断点续传 + 失败重试 + 限流间隔

用法:
  python scripts/download_lst.py --start 2026-01-01 --end 2026-08-25
  python scripts/download_lst.py --start 2026-08-01 --tiles h26v05,h27v05
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

# Earthdata token（会过期，约2个月；过期后到 earthdata 重新生成替换）
TOKEN = os.environ.get("EARTHDATA_TOKEN", "") or _cred("EARTHDATA_TOKEN")

PRODUCT = "MOD11A2"
OUT_DIR = r"C:\vlmdisaster\data\raw\03_thermal_infrared\lst"
BASE = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61"

# 中国核心区 tile（sinusoidal 网格 h##v##）。全中国约 15 个，这里先给核心 8 个
DEFAULT_TILES = ["h23v04", "h24v04", "h25v04", "h26v04", "h27v04",
                 "h25v05", "h26v05", "h27v05"]

HDF4_MAGIC = b"\x0e\x03\x13\x01"
MIN_SIZE = 1_000_000  # 1MB，防假文件/截断


from dl_common import curl_get, http_download


def list_dir_tiles(d):
    """解析某日期的 LAADS 目录页，返回 {tile: 文件名}"""
    doy = d.timetuple().tm_yday
    url = f"{BASE}/{PRODUCT}/{d.year}/{doy:03d}/"
    html = curl_get(url, 60, [f"Authorization: Bearer {TOKEN}"])
    text = html.decode("utf-8", errors="ignore")
    files = {}
    for m in re.finditer(r'href="([^"]+\.hdf)"', text):
        full = m.group(1)
        fn = full.split("/")[-1]  # 目录页 href 是完整 URL，取纯文件名
        mm = re.search(r'h\d{2}v\d{2}', fn)
        if mm:
            files[mm.group(0)] = fn
    return files


def mod11a2_dates(start, end):
    """生成范围内的 8 天合成期日期（儒略日 1,9,17,...）"""
    dates = []
    for y in range(start.year, end.year + 1):
        base = date(y, 1, 1)
        for doy in range(0, 366, 8):
            d = base + timedelta(days=doy)
            if d.year != y:
                break
            if start <= d <= end:
                dates.append(d)
    return dates


def download_file(url, save_path, max_attempts=3):
    """下载单个 HDF，断点续传 + 魔数/大小双重校验"""
    # 已有完整文件 → 跳过
    if os.path.exists(save_path):
        if os.path.getsize(save_path) >= MIN_SIZE:
            with open(save_path, "rb") as f:
                if f.read(4) == HDF4_MAGIC:
                    return "exists"
        os.remove(save_path)  # 残缺或假文件，重下

    tmp = save_path + ".part"
    # 清理假/残留 tmp（登录页等非 HDF 内容不能续传）
    if os.path.exists(tmp):
        with open(tmp, "rb") as f:
            magic = f.read(4)
        if magic != HDF4_MAGIC or os.path.getsize(tmp) < 1000:
            try:
                os.remove(tmp)
            except OSError:
                pass

    return http_download(url, save_path,
                         headers=[f"Authorization: Bearer {TOKEN}"],
                         max_attempts=max_attempts, min_size=MIN_SIZE,
                         magic=HDF4_MAGIC, resume=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期（默认今天）")
    parser.add_argument("--tiles", type=str, default=None,
                        help="逗号分隔 tile，如 h26v05,h27v05（默认中国核心8个）")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    tiles = args.tiles.split(",") if args.tiles else DEFAULT_TILES

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  MODIS LST (MOD11A2) 下载")
    print("=" * 60)
    print(f"  范围: {start} ~ {end}")
    print(f"  tile: {tiles}")
    print(f"  输出: {OUT_DIR}")
    print()

    dates = mod11a2_dates(start, end)
    print(f"  合成期日期共 {len(dates)} 个，最多 {len(dates) * len(tiles)} 个文件")
    print()

    stats = {"ok": 0, "exists": 0, "bad": 0, "fail": 0}
    for d in dates:
        dstr = d.isoformat()
        files = list_dir_tiles(d)
        if not files:
            print(f"  [{dstr}] 目录无数据（合成期可能未发布）", flush=True)
            continue
        for tile in tiles:
            if tile not in files:
                print(f"  [{dstr} {tile}] 目录无此 tile", flush=True)
                continue
            fn = files[tile]
            url = f"{BASE}/{PRODUCT}/{d.year}/{d.timetuple().tm_yday:03d}/{fn}"
            save = os.path.join(OUT_DIR, fn)
            r = download_file(url, save)
            stats[r] += 1
            if r == "ok":
                print(f"  [{dstr} {tile}] ✓ 已下载 ({os.path.getsize(save)/1e6:.1f}MB)", flush=True)
            elif r == "exists":
                print(f"  [{dstr} {tile}] · 已有", flush=True)
            elif r == "bad":
                print(f"  [{dstr} {tile}] ✗ 假文件（非HDF）", flush=True)
            else:
                print(f"  [{dstr} {tile}] ✗ 失败", flush=True)
            time.sleep(0.3)  # 限流（提速）
        time.sleep(0.3)

    print()
    print("=" * 60)
    print(f"  完成: 下载{stats['ok']} 已有{stats['exists']} 假文件{stats['bad']} 失败{stats['fail']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
