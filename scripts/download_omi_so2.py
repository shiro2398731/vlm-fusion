# -*- coding: utf-8 -*-
r"""
OMI SO2 日网格（OMSO2e v004）下载器 —— 火山 SO2 元素用

产品：OMI/Aura Sulfur Dioxide Daily L3 1 day Best Pixel 0.25° V4（OMSO2e）
      全球日网格，约 2.9 MB/天
真值：MSVOLSO2L4 多卫星火山 SO2 数据库（另一个下载器/单文件）
认证：Earthdata token（复用 download_tropomi.py 里的 TOKEN）

用法：
  .\venv\Scripts\python.exe scripts\download_omi_so2.py --start 2020-01-01 --end 2020-01-07
  .\venv\Scripts\python.exe scripts\download_omi_so2.py --start 2020-01-01 --end 2024-12-31
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
OUT = os.path.join(ROOT, "data", "raw", "11_atmospheric", "omi_so2")
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
SHORT = "OMSO2e"
VER = "v004"
MIN = 300_000  # 0.3MB 下限（UVAI 单文件 0.56MB）

S = requests.Session()
S.trust_env = False
H = {"Authorization": "Bearer " + TOKEN}


def list_granules(t0, t1):
    """返回 [(basename, url)]，只取 GET DATA 的 .he5（排除 _mvs.h5）"""
    out = []
    page = 1
    while True:
        r = S.get(CMR, params={"short_name": SHORT,
                               "temporal": "%sT00:00:00Z,%sT00:00:00Z" % (t0, t1),
                               "page_size": 2000, "page_num": page},
                  headers=H, timeout=120)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            break
        for it in items:
            name = (it.get("meta", {}).get("native-id") or "").replace(".he5", "")
            for ru in it.get("umm", {}).get("RelatedUrls", []):
                u = ru.get("URL", "")
                if (ru.get("Type") == "GET DATA" and u.endswith(".he5")
                        and "_mvs" not in u and (VER == "" or VER in os.path.basename(u))):
                    out.append((os.path.basename(u), u))
                    break
        if len(items) < 2000:
            break
        page += 1
    return out


def download(url, dst):
    part = dst + ".part"
    for a in range(1, 4):
        try:
            with S.get(url, headers=H, stream=True, timeout=(30, 300), allow_redirects=True) as r:
                if r.status_code != 200:
                    print("    HTTP %s %s" % (r.status_code, os.path.basename(dst)))
                    return "fail"
                n = 0
                with open(part, "wb") as f:
                    for ch in r.iter_content(1024 * 256):
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
    global SHORT, VER, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--short", default="OMSO2e", help="CMR short_name，如 OSO2e / OMAERUVd")
    ap.add_argument("--version", default="v004", help="文件名版本过滤，空串=不过滤")
    ap.add_argument("--out", default=OUT, help="输出目录")
    a = ap.parse_args()
    SHORT, VER, OUT = a.short, a.version, a.out
    os.makedirs(OUT, exist_ok=True)
    print("OMI SO2 日网格（OMSO2e）%s ~ %s" % (a.start, a.end))
    gl = list_granules(a.start, a.end)
    print("  待下 %d 个文件" % len(gl))
    ok = skip = fail = 0
    for i, (name, url) in enumerate(sorted(gl), 1):
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) >= MIN:
            skip += 1
            continue
        res = download(url, dst)
        if res == "ok":
            ok += 1
        else:
            fail += 1
        if i % 50 == 0 or i == len(gl):
            print("  [%d/%d] ok=%d skip=%d fail=%d" % (i, len(gl), ok, skip, fail))
    print("\n完成: ok=%d skip=%d fail=%d -> %s" % (ok, skip, fail, OUT))


if __name__ == "__main__":
    main()
