# -*- coding: utf-8 -*-
r"""
MOD13C2 NDVI 月产品下载器 —— 干旱×光学元素用

产品：MODIS/Terra Vegetation Indices Monthly L3 Global 0.05Deg CMG（MOD13C2 v061）
      全球 0.05° 月合成，约 90 MB/月
认证：Earthdata token（复用 download_tropomi.py 里的 TOKEN）

用法：
  .\venv\Scripts\python.exe scripts\download_modis_ndvi.py --start 2024-01 --end 2024-03
  .\venv\Scripts\python.exe scripts\download_modis_ndvi.py --start 2020-01 --end 2024-12
"""
import argparse
import os
import sys
import time
from datetime import date

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_tropomi import TOKEN  # noqa: E402

ROOT = r"C:\vlmdisaster"
OUT = os.path.join(ROOT, "data", "raw", "01_optical_vnir", "modis_ndvi")
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
SHORT = "MOD13C2"
MIN = 30_000_000  # 正常约 90MB，30MB 作下限

S = requests.Session()
S.trust_env = False
H = {"Authorization": "Bearer " + TOKEN}


def month_bounds(y, m):
    if m == 12:
        return date(y, 12, 1), date(y + 1, 1, 1)
    return date(y, m, 1), date(y, m + 1, 1)


def list_granules(y, m):
    t0, t1 = month_bounds(y, m)
    r = S.get(CMR, params={"short_name": SHORT,
                           "temporal": "%sT00:00:00Z,%sT00:00:00Z" % (t0, t1),
                           "page_size": 100},
              headers=H, timeout=120)
    r.raise_for_status()
    out = []
    for it in r.json().get("items", []):
        for ru in it.get("umm", {}).get("RelatedUrls", []):
            u = ru.get("URL", "")
            if ru.get("Type") == "GET DATA" and u.endswith(".hdf"):
                out.append((os.path.basename(u), u))
                break
    return out


def download(url, dst):
    part = dst + ".part"
    for a in range(1, 4):
        try:
            with S.get(url, headers=H, stream=True, timeout=(30, 600), allow_redirects=True) as r:
                if r.status_code != 200:
                    print("    HTTP %s %s" % (r.status_code, os.path.basename(dst)))
                    return "fail"
                n = 0
                with open(part, "wb") as f:
                    for ch in r.iter_content(1024 * 512):
                        f.write(ch)
                        n += len(ch)
            if n >= MIN:
                os.replace(part, dst)
                return "ok"
            print("    太小 %d B" % n)
        except Exception as e:
            print("    异常 %s" % str(e)[:80])
            time.sleep(3)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    a = ap.parse_args()
    y0, m0 = [int(x) for x in a.start.split("-")]
    y1, m1 = [int(x) for x in a.end.split("-")]
    os.makedirs(OUT, exist_ok=True)

    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1

    print("MODIS NDVI 月产品（MOD13C2）%04d-%02d ~ %04d-%02d，共 %d 个月" % (y0, m0, y1, m1, len(months)))
    ok = skip = fail = 0
    for i, (yy, mm) in enumerate(months, 1):
        gl = list_granules(yy, mm)
        if not gl:
            print("  %04d-%02d: 无 granule" % (yy, mm))
            fail += 1
            continue
        name, url = gl[0]
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) >= MIN:
            skip += 1
            continue
        res = download(url, dst)
        if res == "ok":
            ok += 1
        else:
            fail += 1
        print("  [%d/%d] %04d-%02d %s %s" % (i, len(months), yy, mm, res, name))
    print("\n完成: ok=%d skip=%d fail=%d -> %s" % (ok, skip, fail, OUT))


if __name__ == "__main__":
    main()
