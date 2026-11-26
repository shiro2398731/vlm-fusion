# -*- coding: utf-8 -*-
r"""
原始影像工作流跑批：CDSE 搜索 → 下载 → 渲染 → 判读 → 对照。

用法：
  python scripts/raw_scene_runner.py --tag sindh_0827 --point 68.4,27.0 \
      --start 2025-08-26 --end 2025-08-28 --windows 6

输出：
  data/raw/scenes/<tag>/            SAFE
  outputs/raw_test_<tag>/           渲染窗口 + windows.json
  outputs/raw_test_<tag>/raw_workflow_result.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import zipfile
from urllib.parse import quote

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROXY = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
ROOT = r"C:\vlmdisaster"
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")


def token():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from credentials_local import CDSE_USER as U, CDSE_PASS as P
    except ImportError:
        from config_local import CDSE_USER as U, CDSE_PASS as P
    r = requests.post(TOKEN_URL, data={"grant_type": "password", "username": U,
                                       "password": P, "client_id": "cdse-public"},
                      proxies=PROXY, timeout=60)
    return r.json()["access_token"]


def search(tok, start, end, point, max_cloud=20, top=20):
    lon, lat = point
    filt = ["Collection/Name eq 'SENTINEL-2'", "contains(Name,'MSIL2A')",
            "ContentDate/Start ge %sT00:00:00.000Z" % start,
            "ContentDate/Start le %sT23:59:59.000Z" % end,
            "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            "and att/OData.CSC.DoubleAttribute/Value le %d)" % max_cloud,
            "OData.CSC.Intersects(area=geography'SRID=4326;POINT(%s %s)')" % (lon, lat)]
    enc = quote(" and ".join(filt), safe="()=/,:")
    url = "%s?$filter=%s&$orderby=ContentDate/Start desc&$top=%d" % (ODATA, enc, top)
    r = requests.get(url, headers={"Authorization": "Bearer " + tok}, proxies=PROXY, timeout=120)
    r.raise_for_status()
    return r.json().get("value", [])


def download(tok, pid, dst):
    url = "%s(%s)/$value" % (ODATA, pid)
    r = requests.get(url, headers={"Authorization": "Bearer " + tok}, proxies=PROXY,
                     timeout=120, allow_redirects=False)
    loc = r.headers.get("Location")
    if not loc:
        raise RuntimeError("no redirect: %s" % r.status_code)
    t0 = time.time()
    with requests.get(loc, headers={"Authorization": "Bearer " + tok}, proxies=PROXY,
                      stream=True, timeout=1800) as resp:
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(16 * 1024 * 1024):
                f.write(chunk)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--point", required=True, help="lon,lat")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--max-cloud", type=float, default=20)
    ap.add_argument("--api", default="http://127.0.0.1:8078")
    args = ap.parse_args()

    lon, lat = [float(x) for x in args.point.split(",")]
    work = os.path.join(ROOT, "data", "raw", "scenes", args.tag)
    os.makedirs(work, exist_ok=True)

    tok = token()
    val = search(tok, args.start, args.end, (lon, lat), max_cloud=args.max_cloud)
    if not val:
        raise SystemExit("没搜到产品")
    # 选体积最小的（下载快）
    val = sorted(val, key=lambda it: it.get("ContentLength") or 9e9)
    it = val[0]
    name = it["Name"]
    print("选定：%s  %.0f MB" % (name, (it.get("ContentLength") or 0) / 1e6))

    zip_path = os.path.join(work, name + ".zip")
    if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 200e6):
        sec = download(tok, it["Id"], zip_path)
        print("下载完成 %.0f MB，%.0fs" % (os.path.getsize(zip_path) / 1e6, sec))

    # 解压
    safe_dirs = sorted(glob.glob(os.path.join(work, "*.SAFE")))
    if not safe_dirs:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(work)
        safe_dirs = sorted(glob.glob(os.path.join(work, "*.SAFE")))
    safe = safe_dirs[0]
    print("SAFE:", os.path.basename(safe))

    outdir = os.path.join(ROOT, "outputs", "raw_test_%s" % args.tag)
    cmd = [PY, "-u", os.path.join(ROOT, "scripts", "raw_s2_to_ndwi.py"),
           "--safe", safe, "--windows", str(args.windows), "--outdir", outdir]
    subprocess.run(cmd, check=True)

    cmd2 = [PY, "-u", os.path.join(ROOT, "scripts", "validate_raw_workflow.py"),
            "--dir", outdir, "--api", args.api]
    subprocess.run(cmd2, check=True)
    print("完成：", outdir)


if __name__ == "__main__":
    main()
