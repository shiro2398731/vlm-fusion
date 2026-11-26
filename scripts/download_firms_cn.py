# -*- coding: utf-8 -*-
r"""
FIRMS 火点下载（中国区域，VIIRS SNPP 标准处理）

源：NASA FIRMS area API
  产品：VIIRS_SNPP_SP（标准处理，有历史）
  区域：CHN（中国）
  每天一个 CSV

用法:
  python download_firms_cn.py --start 2024-01-01 --end 2024-12-31
"""
import os
import sys
import time
import argparse
from datetime import date, timedelta
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
S = requests.Session(); S.trust_env = False

KEY = "534f765bf2cfa1ca794fbdd531e61d90"
OUT = r"C:\vlmdisaster\data\raw\03_thermal_infrared\firms\chn"
MIN = 100
AREA = "73,18,135,54"      # 中国 bbox: west,south,east,north


def dl(day):
    fn = "fire_%s.csv" % day.isoformat()
    dst = os.path.join(OUT, fn)
    if os.path.exists(dst) and os.path.getsize(dst) >= MIN:
        return "skip"
    u = ("https://firms.modaps.eosdis.nasa.gov/api/area/csv/%s/VIIRS_SNPP_SP/%s/1/%s"
         % (KEY, AREA, day.isoformat()))
    part = dst + ".part"
    for a in range(1, 4):
        try:
            r = S.get(u, timeout=a * 120)
            if r.status_code != 200:
                print("    HTTP %d" % r.status_code); time.sleep(5); continue
            txt = r.text
            if not txt.startswith("latitude"):
                print("    bad header: %s" % txt[:60]); return "bad"
            with open(part, "w", encoding="utf-8") as f:
                f.write(txt)
            if os.path.getsize(part) >= MIN:
                os.replace(part, dst)
                return "ok"
        except Exception as e:
            print("    %d %s" % (a, str(e)[:60])); time.sleep(5)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    print("待下载 %d 天（中国区域）" % len(days))

    ok = skip = fail = 0
    d = d0
    while d <= d1:
        res = dl(d)
        if res == "ok": ok += 1
        elif res == "skip": skip += 1
        else: fail += 1; print("  %s %s" % (res, d))
        if (ok + skip) % 30 == 0:
            print("  ...%s  ok=%d skip=%d" % (d, ok, skip), flush=True)
        time.sleep(0.4)
        d += timedelta(days=1)
    print("\n完成: ok=%d skip=%d fail=%d" % (ok, skip, fail))


if __name__ == "__main__":
    main()
