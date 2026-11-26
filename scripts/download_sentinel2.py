# -*- coding: utf-8 -*-
"""
Sentinel-2 L1C 光学下载脚本 — Copernicus 质量优先版 v1

用途: 光学 VNIR 源（源1）——替代 DisasterM3 的半成品，自采真原始
产品: Sentinel-2 L1C（MSIL1C，大气顶反射率，后续自己辐射定标/大气校正）
数据源: Copernicus Data Space Ecosystem（CDSE，OData + OAuth token）

设计要点:
  - 搜索: OData Products 接口，按 AOI(点) + 时间窗 + 云量上限过滤
  - 下载: OAuth2 password grant 拿 access token → OData /Products(id)/$value 下 zip
          curl 必须带 --location-trusted（否则重定向后 Bearer 头会丢）
  - 质量校验: zip 魔数 PK\\x03\\x04 + 大小下限
  - 磁盘预算: --max-gb 闸门（默认 15GB）
  - 自检: --dry-run 只搜索列清单

认证: 需要你的 Copernicus 账号（注册邮箱 + 密码）。密码不写死在本脚本里，
      通过 --user / --pass 传入，或设环境变量 CDSE_USER / CDSE_PASS。
      （比 Earthdata 的 token 敏感，别提交到任何仓库）

依赖: 只用 Python 标准库 + curl.exe（无需 boto3 / sentinelsat）

用法:
  # 1) 先自检（需要账号，dry-run 只搜索）
  python scripts/download_sentinel2.py --user 你的邮箱 --pass 你的密码 ^
      --start 2026-08-01 --end 2026-08-31 --point 116.4,39.9 --max-cloud 20 --dry-run

  # 2) 确认后真下
  python scripts/download_sentinel2.py --user 你的邮箱 --pass 你的密码 ^
      --start 2026-08-01 --end 2026-08-31 --point 116.4,39.9 --max-cloud 20

  # point: 经度,纬度（AOI 中心点）；也可换 --bbox 西,南,东,北
"""
import os
from config_local import cred as _cred
import sys
import time
import json
import argparse
import subprocess
from datetime import date

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2"
ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

ZIP_MAGIC = b"PK\x03\x04"
MIN_SIZE = 100_000_000  # 100MB 下限（S2 L1C 单景通常 500MB~1GB）


def curl(args, timeout=1800):
    base = ["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke"]
    try:
        return subprocess.run(base + args, timeout=timeout, capture_output=True)
    except Exception as e:
        print(f"    curl 异常: {e}")
        return None


def get_token(user, password):
    r = curl(["-s", "-X", "POST", TOKEN_URL,
              "-H", "Content-Type: application/x-www-form-urlencoded",
              "--data-urlencode", "grant_type=password",
              "--data-urlencode", f"username={user}",
              "--data-urlencode", f"password={password}",
              "--data-urlencode", "client_id=cdse-public"], timeout=120)
    if r is None:
        return None
    try:
        return json.loads(r.stdout.decode("utf-8", errors="ignore")).get("access_token")
    except Exception:
        return None


def search(token, start, end, point=None, bbox=None, max_cloud=100, top=100):
    """OData 搜索 S2 L1C，返回 [(Id, Name, size)]"""
    filt = ["Collection/Name eq 'SENTINEL-2'",
            "contains(Name,'MSIL1C')",
            f"ContentDate/Start ge {start}T00:00:00.000Z",
            f"ContentDate/Start le {end}T23:59:59.000Z",
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"]
    if point:
        lon, lat = point
        geo = f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})')"
    else:
        w, s, e, n = bbox
        geo = (f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({w} {s},{e} {s},"
               f"{e} {n},{w} {n},{w} {s}))')")
    filt.append(geo)
    # 关键：OData 过滤式含空格/单引号，必须 URL 编码（空格→%20，'→%27 等），
    # 否则 curl/python 都会因非法字符发不出去（表现为 000 / InvalidURL / 搜不到）。
    from urllib.parse import quote
    raw_filter = " and ".join(filt)
    enc_filter = quote(raw_filter, safe="()=/,:")
    url = (f"{ODATA}?$filter={enc_filter}"
           f"&$orderby=ContentDate/Start&$top={top}")
    r = curl(["-s", "-H", f"Authorization: Bearer {token}", url], timeout=120)
    if r is None:
        return []
    try:
        data = json.loads(r.stdout.decode("utf-8", errors="ignore"))
    except Exception:
        return []
    out = []
    for it in data.get("value", []):
        pid = it.get("Id")
        name = it.get("Name", "")
        sz = it.get("ContentLength") or 0
        if pid and name:
            out.append((pid, name, int(sz) if str(sz).isdigit() else 0))
    return out


def download_file(token, pid, save_path, max_attempts=3):
    if os.path.exists(save_path):
        if os.path.getsize(save_path) >= MIN_SIZE:
            with open(save_path, "rb") as f:
                if f.read(4) == ZIP_MAGIC:
                    return "exists", os.path.getsize(save_path)
        os.remove(save_path)

    tmp = save_path + ".part"
    hdr = save_path + ".hdr.txt"
    url = f"{ODATA}({pid})/$value"
    for attempt in range(1, max_attempts + 1):
        cmd = ["--location-trusted",
               "-H", f"Authorization: Bearer {token}",
               "--connect-timeout", "30", "--max-time", "3600",
               "-D", hdr,          # 存响应头，便于诊断
               "-C", "-", "-s", "-o", tmp, url]
        curl(cmd, timeout=3660)
        if os.path.exists(tmp) and os.path.getsize(tmp) >= MIN_SIZE:
            with open(tmp, "rb") as f:
                magic = f.read(4)
            if magic == ZIP_MAGIC:
                ok = False
                for _ in range(10):
                    try:
                        os.replace(tmp, save_path)
                        ok = True
                        break
                    except PermissionError:
                        time.sleep(2)
                if ok:
                    return "ok", os.path.getsize(save_path)
            # 不是 zip：保留文件 + 打印头 4 字节（不删，便于诊断）
            print(f"    ! 非zip文件 magic={magic.hex()} size={os.path.getsize(tmp)} (保留为 {os.path.basename(tmp)})")
            return "bad", os.path.getsize(tmp)
            return "bad", 0
        time.sleep(10)
    return "fail", 0


def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=_cred("CDSE_USER"), help="Copernicus 账号邮箱")
    ap.add_argument("--pass", dest="password", default=_cred("CDSE_PASS"), help="Copernicus 密码")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--point", help="AOI 中心点 经度,纬度")
    ap.add_argument("--bbox", help="AOI 包围盒 西,南,东,北（与 --point 二选一）")
    ap.add_argument("--max-cloud", type=float, default=100, help="云量上限 %%（默认100）")
    ap.add_argument("--max-gb", type=float, default=15.0, help="磁盘预算上限 GB（默认15）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.user or not args.password:
        print("✗ 需要 Copernicus 账号：--user 邮箱 --pass 密码（或设 CDSE_USER/CDSE_PASS）")
        sys.exit(1)
    if not args.point and not args.bbox:
        print("✗ 需要 --point 经度,纬度 或 --bbox 西,南,东,北")
        sys.exit(1)

    point = [float(x) for x in args.point.split(",")] if args.point else None
    bbox = [float(x) for x in args.bbox.split(",")] if args.bbox else None

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Sentinel-2 L1C 下载 (Copernicus) v1")
    print("=" * 60)
    print(f"  时间窗: {args.start} ~ {args.end}")
    print(f"  AOI   : {'point ' + str(point) if point else 'bbox ' + str(bbox)}")
    print(f"  云量  : ≤ {args.max_cloud}%")
    print(f"  输出  : {OUT_DIR}")
    print(f"  预算  : {args.max_gb} GB" + ("   [DRY-RUN]" if args.dry_run else ""))
    print()

    print("  取 access token...", flush=True)
    token = get_token(args.user, args.password)
    if not token:
        print("  ✗ 获取 token 失败（账号/密码错，或 CDSE 身份服务不通）")
        return

    print("  搜索中...", flush=True)
    prods = search(token, args.start, args.end, point, bbox, args.max_cloud)
    if not prods:
        print("  未搜到产品（检查 AOI/时间窗/云量；也可能 OData 不通）")
        return

    total = sum(p[2] for p in prods)
    print(f"  搜到 {len(prods)} 景，合计约 {human(total)}")
    print()
    for pid, name, sz in prods:
        print(f"    · {name}  ({human(sz) if sz else '未知'})")
    print()

    if args.dry_run:
        print("  [DRY-RUN] 到此为止，未下载。确认清单后去掉 --dry-run 再跑。")
        return

    budget = args.max_gb * 1024 ** 3
    stats = {"ok": 0, "exists": 0, "bad": 0, "fail": 0}
    used = 0
    for pid, name, sz in prods:
        save = os.path.join(OUT_DIR, name)
        r, got = download_file(token, pid, save)
        stats[r] += 1
        used += got
        tag = {"ok": "✓ 已下载", "exists": "· 已有", "bad": "✗ 假文件", "fail": "✗ 失败"}[r]
        print(f"  [{name}] {tag} ({human(got)})" if got else f"  [{name}] {tag}", flush=True)
        if used >= budget:
            print(f"\n  ⚠️ 已达磁盘预算 {args.max_gb}GB，停止（剩余下次再下）")
            break
        time.sleep(3)

    print()
    print("=" * 60)
    print(f"  完成: 下载{stats['ok']} 已有{stats['exists']} 假文件{stats['bad']} 失败{stats['fail']}")
    print(f"  本次占用约 {human(used)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
