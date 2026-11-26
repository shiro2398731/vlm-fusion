# -*- coding: utf-8 -*-
r"""
Sentinel-2 光学下载（requests 版，支持 L2A / L1C）—— CDSE OData + OAuth

改动说明（2026-10-18）：
  - 本机 curl 的 schannel 坏了（SEC_E_NO_CREDENTIALS），全部改 requests
  - 新增 --product L2A|L1C，默认 **L2A**（已大气校正，值可比、适合训练）
  - 新增 --max-bytes 总预算闸门

用法:
  python download_sentinel2.py --user 邮箱 --pass 密码 --start 2024-07-01 --end 2024-07-31 \
      --point 120.5,31.5 --max-cloud 20 --dry-run

  python download_sentinel2.py ... --product L2A
"""
import os
from config_local import cred as _cred
import sys
import json
import time
import argparse
from datetime import date
import requests
from urllib.parse import quote

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2"
ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ZIP_MAGIC = b"PK\x03\x04"
MIN_SIZE = 50_000_000

S = requests.Session()
S.trust_env = False


def get_token(user, password):
    for a in range(1, 6):
        try:
            r = S.post(TOKEN_URL, data={
                "grant_type": "password",
                "username": user,
                "password": password,
                "client_id": "cdse-public",
            }, timeout=60)
            if r.status_code != 200:
                print("  取 token 失败 HTTP", r.status_code, r.text[:200])
                if r.status_code == 401:
                    return None
            else:
                return r.json().get("access_token")
        except Exception as e:
            print("  取 token 异常(%d/5): %s" % (a, str(e)[:70]))
        time.sleep(5 * a)
    return None


def search(token, start, end, product="L2A", point=None, bbox=None, max_cloud=100, top=100):
    tag = "MSIL2A" if product.upper() == "L2A" else "MSIL1C"
    filt = ["Collection/Name eq 'SENTINEL-2'",
            f"contains(Name,'{tag}')",
            f"ContentDate/Start ge {start}T00:00:00.000Z",
            f"ContentDate/Start le {end}T23:59:59.000Z",
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"]
    if point:
        lon, lat = point
        geo = f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})')"
    else:
        w, s_, e, n = bbox
        geo = (f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({w} {s_},{e} {s_},"
               f"{e} {n},{w} {n},{w} {s_}))')")
    filt.append(geo)
    enc = quote(" and ".join(filt), safe="()=/,:")
    url = f"{ODATA}?$filter={enc}&$orderby=ContentDate/Start&$top={top}"
    for a in range(1, 4):
        try:
            r = S.get(url, headers={"Authorization": "Bearer " + token}, timeout=(30, 180))
            if r.status_code != 200:
                print("  搜索 HTTP", r.status_code, r.text[:150]); time.sleep(3); continue
            out = []
            for it in r.json().get("value", []):
                pid = it.get("Id"); name = it.get("Name", "")
                sz = it.get("ContentLength") or 0
                if pid and name:
                    out.append((pid, name, int(sz) if str(sz).isdigit() else 0))
            return out
        except Exception as e:
            print("  搜索异常:", str(e)[:80]); time.sleep(3)
    return []


def download(token, pid, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) >= MIN_SIZE:
        with open(save_path, "rb") as f:
            if f.read(4) == ZIP_MAGIC:
                return "exists"
    part = save_path + ".part"
    url = f"{ODATA}({pid})/$value"
    hdrs = {"Authorization": "Bearer " + token}

    def open_stream(u, h):
        """手动跟随重定向，每跳保留 Authorization（requests 会自动剥离）"""
        for _ in range(10):
            r = S.get(u, headers=h, stream=True, timeout=(60, 3600), allow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("Location")
                r.close()
                if not loc:
                    return None
                u = loc if loc.startswith("http") else requests.compat.urljoin(u, loc)
                continue
            return r
        return None

    for a in range(1, 4):
        try:
            existing = os.path.getsize(part) if os.path.exists(part) else 0
            h = dict(hdrs)
            if existing:
                h["Range"] = "bytes=%d-" % existing
            r = open_stream(url, h)
            if r is None:
                print("    重定向异常"); time.sleep(10); continue
            if r.status_code == 416:
                os.replace(part, save_path); return "ok"
            if r.status_code != 200 and r.status_code != 206:
                if r.status_code == 401:
                    r.close()
                    print("    token 过期，需要刷新")
                    return "expired"
                print("    下载 HTTP", r.status_code, r.text[:120]); r.close(); time.sleep(10); continue
            if existing and r.status_code == 200:
                existing = 0          # 服务端不支续传，从头写
            n = existing
            mode = "ab" if existing else "wb"
            with open(part, mode) as f:
                for ch in r.iter_content(1024 * 1024):
                    f.write(ch); n += len(ch)
                    if n % (200 * 1024 * 1024) < 1024 * 1024:
                        print("      %.0f MB" % (n / 1e6), flush=True)
            if n >= MIN_SIZE:
                with open(part, "rb") as f:
                    if f.read(4) == ZIP_MAGIC:
                        os.replace(part, save_path)
                        return "ok"
                os.remove(part)
                return "bad"
        except Exception as e:
            print("    下载异常:", str(e)[:80]); time.sleep(10)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=_cred("CDSE_USER"))
    ap.add_argument("--pass", dest="pwd", default=_cred("CDSE_PASS"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--product", default="L2A", choices=["L2A", "L1C"])
    ap.add_argument("--point", help="经度,纬度")
    ap.add_argument("--bbox", help="西,南,东,北")
    ap.add_argument("--max-cloud", type=float, default=20)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--max-gb", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.user or not args.pwd:
        print("需要 --user / --pass（CDSE 账号）"); return

    point = tuple(float(x) for x in args.point.split(",")) if args.point else None
    bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None

    print("=== Sentinel-2 %s 下载 ===" % args.product)
    print("  时间: %s ~ %s | 云量 <= %.0f%% | 预算 %.0f GB" % (args.start, args.end, args.max_cloud, args.max_gb))

    token = get_token(args.user, args.pwd)
    if not token:
        print("认证失败"); return
    print("  ✅ 认证成功")

    items = search(token, args.start, args.end, args.product, point, bbox, args.max_cloud, args.top)
    print("  搜到 %d 景" % len(items))
    if not items:
        return
    total = sum(sz for _, _, sz in items)
    print("  总大小 %.2f GB" % (total / 1e9))
    for pid, name, sz in items[:10]:
        print("    %-70s %.0f MB" % (name[:68], sz / 1e6))
    if args.dry_run:
        print("\n(dry-run，不下载)"); return

    os.makedirs(OUT_DIR, exist_ok=True)
    budget = args.max_gb * 1e9
    used = 0
    ok = skip = fail = 0
    for pid, name, sz in items:
        if used + sz > budget:
            print("  达到预算上限，停止"); break
        dst = os.path.join(OUT_DIR, name if name.endswith(".zip") else name + ".zip")
        res = download(token, pid, dst)
        if res == "expired":
            print("  刷新 token 重试...")
            token = get_token(args.user, args.pwd)
            if not token:
                print("  刷新失败，停止"); break
            res = download(token, pid, dst)
        if res == "expired":
            token = get_token(args.user, args.pwd)
            if token:
                res = download(token, pid, dst)
        if res == "ok": ok += 1; used += sz; print("  ok %s" % name[:70])
        elif res == "exists": skip += 1; used += sz
        else: fail += 1; print("  %s %s" % (res, name[:70]))
    print("\n完成: ok=%d skip=%d fail=%d | 已用 %.2f GB" % (ok, skip, fail, used / 1e9))


if __name__ == "__main__":
    main()
