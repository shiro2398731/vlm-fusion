# -*- coding: utf-8 -*-
r"""
CMEMS 通用下载器（S3 匿名直链）—— 支持任一 CMEMS 产品。

原理：STAC 找产品 → S3 列文件(?list-type=2&prefix=) → requests 直下 .nc。
无需 token。

用法:
  python download_cmems.py --product <PID> --dataset <DATASET_SUFFIX> --start 2024-01-01 --end 2024-01-05
  python download_cmems.py --product SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009 --dataset OSISAF-GLO-SEAICE_CONC_CONT_TIMESERIES-NH-LA-OBS_202003 --start 2021-01-01 --end 2021-01-10
  python download_cmems.py --list-products WAVE   # 按关键词列产品
"""
import os, re, sys, time, argparse
from datetime import date, timedelta
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 绕开代理（本机代理会拖慢/卡 S3 握手）
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.proxies = {}

STAC_ROOT = "https://s3.waw3-1.cloudferro.com/mdl-metadata/metadata/"
MIN = 100_000


def get_native_prefix(product, dataset):
    """从 STAC 拿 native 资产的 S3 前缀"""
    u = "%s%s/%s/dataset.stac.json" % (STAC_ROOT, product, dataset)
    for attempt in range(1, 5):
        try:
            r = SESSION.get(u, timeout=(30, 120))
            if r.status_code != 200:
                raise RuntimeError("dataset.stac.json 404: %s" % u)
            d = r.json()
            nat = (d.get("assets") or {}).get("native", {}).get("href")
            if not nat:
                raise RuntimeError("无 native 资产")
            return nat
        except Exception as e:
            print("  STAC 重试 %d: %s" % (attempt, str(e)[:70])); time.sleep(5)
    raise RuntimeError("STAC 拉取失败: %s" % u)


def split_s3(url):
    """https://s3.waw3-1.cloudferro.com/<bucket>/<key...> -> (endpoint, bucket, key)"""
    m = re.match(r"(https://[^/]+)/([^/]+)/(.+)", url)
    return m.group(1), m.group(2), m.group(3)


def list_keys(endpoint, bucket, prefix):
    keys = []
    token = None
    while True:
        u = "%s/%s?list-type=2&prefix=%s&max-keys=1000" % (endpoint, bucket, prefix)
        if token:
            u += "&continuation-token=" + requests.utils.quote(token, safe="")
        r = None
        for a in range(1, 6):
            try:
                r = SESSION.get(u, timeout=(30, 120))
                break
            except Exception as e:
                print("    list 重试 %d: %s" % (a, str(e)[:60])); time.sleep(5)
        if r is None or r.status_code != 200:
            break
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if m and "<IsTruncated>true</IsTruncated>" in r.text:
            token = m.group(1)
        else:
            break
    return keys


def download(endpoint, bucket, key, out_dir):
    fn = key.split("/")[-1]
    dst = os.path.join(out_dir, fn)
    if os.path.exists(dst) and os.path.getsize(dst) >= MIN:
        return "skip"
    part = dst + ".part"
    for a in range(1, 4):
        try:
            r = SESSION.get("%s/%s/%s" % (endpoint, bucket, key), stream=True, timeout=(30, 600))
            if r.status_code == 404:
                return "404"
            r.raise_for_status()
            n = 0
            with open(part, "wb") as f:
                for ch in r.iter_content(1024 * 1024):
                    f.write(ch); n += len(ch)
            if n >= MIN:
                os.replace(part, dst)
                return "ok"
        except Exception as e:
            print("    %d 异常: %s" % (a, str(e)[:70])); time.sleep(3)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product")
    ap.add_argument("--dataset", help="dataset_id（STAC dataset.stac.json 的目录名）")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out")
    ap.add_argument("--list-products", help="按关键词列 STAC 产品")
    ap.add_argument("--list-datasets", help="列某产品的 dataset")
    args = ap.parse_args()

    if args.list_products:
        d = SESSION.get(STAC_ROOT + "catalog.stac.json", timeout=30).json()
        kw = args.list_products.upper()
        for lk in d.get("links", []):
            h = lk.get("href", "").split("/")[0]
            if kw in h.upper():
                print("  ", h)
        return

    if args.list_datasets:
        prod = SESSION.get("%s%s/product.stac.json" % (STAC_ROOT, args.list_datasets), timeout=30).json()
        for lk in prod.get("links", []):
            if lk.get("rel") == "item":
                print("  ", lk["href"].split("/")[0])
        return

    if not (args.product and args.dataset and args.start and args.end):
        print("需 --product --dataset --start --end"); return

    native = get_native_prefix(args.product, args.dataset)
    ep, bucket, prefix = split_s3(native)
    base_prefix = prefix.rsplit("/", 1)[0]  # 去掉 dataset id 段？不，prefix 本身就是 dataset 目录
    print("native:", native)
    print("endpoint=%s bucket=%s" % (ep, bucket))

    out = args.out or os.path.join(r"C:\vlmdisaster\data\raw", args.product)
    os.makedirs(out, exist_ok=True)

    d0 = date.fromisoformat(args.start); d1 = date.fromisoformat(args.end)
    # 按月列（多数产品按 年/月/ 或 年/ 分目录）
    months = set()
    d = d0
    while d <= d1:
        months.add((d.year, d.month)); d += timedelta(days=1)
    wanted = set()
    d = d0
    while d <= d1:
        wanted.add(d.strftime("%Y%m%d")); d += timedelta(days=1)

    ok = skip = fail = 0
    found = 0
    for (y, mo) in sorted(months):
        for sub in ("%s/%04d/%02d/" % (prefix, y, mo), "%s/%04d/" % (prefix, y), "%s/" % prefix):
            keys = list_keys(ep, bucket, sub)
            if not keys:
                continue
            for k in keys:
                fn = k.split("/")[-1]
                # 匹配日期
                hit = any(w in fn for w in wanted)
                if not hit:
                    continue
                found += 1
                res = download(ep, bucket, k, out)
                if res == "ok": ok += 1; print("  ok", fn)
                elif res == "skip": skip += 1
                else: fail += 1; print("  %s %s" % (res, fn))
            break  # 这个月找到了就不试其他子路径
    print("\n匹配 %d 文件 | ok=%d skip=%d fail=%d -> %s" % (found, ok, skip, fail, out))


if __name__ == "__main__":
    main()
