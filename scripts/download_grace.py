# -*- coding: utf-8 -*-
r"""
GRACE 地下水/土壤湿度（GRACEDADM_CLSM025GL_7D）下载 —— GES DISC，Earthdata token。
0.25°，7 天一个文件，2.1MB/文件。

用法: python download_grace.py --start 2003-02-03 --end 2024-12-31
      python download_grace.py --start 2024-01-01 --end 2024-12-31
"""
import os, sys, time, argparse
from datetime import date, timedelta
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
S = requests.Session(); S.trust_env = False
TOKEN = open(r'C:\vlmdisaster\scripts\download_lst.py', encoding='utf-8').read().split('TOKEN = "')[1].split('"')[0]
OUT = r"C:\vlmdisaster\data\raw\09_gravity\grace"
BASE = "https://data.gesdisc.earthdata.nasa.gov/data/GRACEDA/GRACEDADM_CLSM025GL_7D.3.0"
MIN = 500_000
HDF_MAGIC = b"\x89HDF"


def dl(d, fn):
    dst = os.path.join(OUT, fn)
    if os.path.exists(dst) and os.path.getsize(dst) >= MIN:
        return "skip"
    u = "%s/%d/%s" % (BASE, d.year, fn)
    part = dst + ".part"
    for a in range(1, 5):
        try:
            r = S.get(u, headers={"Authorization": "Bearer " + TOKEN},
                      timeout=(30, 300), stream=True)
            if r.status_code == 404:
                return "404"
            r.raise_for_status()
            n = 0
            with open(part, "wb") as f:
                for ch in r.iter_content(1024 * 512):
                    f.write(ch); n += len(ch)
            if n >= MIN:
                with open(part, "rb") as f:
                    if f.read(4) != HDF_MAGIC:
                        os.remove(part); return "bad"
                os.replace(part, dst)
                return "ok"
        except Exception as e:
            print("    %d %s" % (a, str(e)[:60])); time.sleep(4)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d0 = date.fromisoformat(args.start); d1 = date.fromisoformat(args.end)
    print("GRACE 下载 %s ~ %s" % (d0, d1))

    # 产品是 7 天一个，日期为每周一
    d = d0
    # 对齐到周一
    while d.weekday() != 0:
        d += timedelta(days=1)
    ok = skip = fail = miss = 0
    while d <= d1:
        fn = "GRACEDADM_CLSM025GL_7D.A%s.030.nc4" % d.strftime("%Y%m%d")
        res = dl(d, fn)
        if res == "ok": ok += 1; print("  ok %s" % fn)
        elif res == "skip": skip += 1
        elif res == "404": miss += 1
        else: fail += 1; print("  %s %s" % (res, fn))
        d += timedelta(days=7)
    print("\n完成: ok=%d skip=%d 404=%d fail=%d" % (ok, skip, miss, fail))


if __name__ == "__main__":
    main()
