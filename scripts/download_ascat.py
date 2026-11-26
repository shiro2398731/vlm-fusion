# -*- coding: utf-8 -*-
r"""
ASCAT 散射计海面风场 · 下载

清单：data\raw\07_scatterometer\ascat_manifest.json（probe_ascat.py 生成）
输出：data\raw\07_scatterometer\raw\<title>

认证
----
  NASA PO.DAAC 走 Earthdata。先把 granule 的 data 链接取出来，
  带 Basic Auth（账号密码）下载；若 401，改走 Earthdata token。

踩坑记录（沿用本项目既有经验）
--------------------------
  1) 自动跳转会剥掉 Authorization 头，必须 allow_redirects=False 手动逐跳
  2) 跳到 S3 / cloudfront 后要主动去掉 Authorization
  3) chunk 用 8 MB、timeout (120, 900)

用法
----
  python scripts/download_ascat.py --probe      # 先只看链接格式，不下
  python scripts/download_ascat.py
  python scripts/download_ascat.py --workers 2
"""
import argparse
from config_local import cred as _cred
import json
import os
import sys
import threading
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

USER = os.environ.get("EARTHDATA_USER", "") or _cred("EARTHDATA_USER")
PWD = os.environ.get("EARTHDATA_PASS", "") or _cred("EARTHDATA_PASS")

BASE = r"C:\vlmdisaster\data\raw\07_scatterometer"
MANIFEST = os.path.join(BASE, "ascat_manifest.json")
OUTDIR = os.path.join(BASE, "raw")

CMR_GRANULE = "https://cmr.earthdata.nasa.gov/search/granules.json"
CHUNK = 8 * 1024 * 1024
TIMEOUT = (120, 900)

_lock = threading.Lock()
_done = {"ok": 0, "fail": 0, "bytes": 0}
_failed = []


def get_data_url(granule_id):
    """从 CMR 取该 granule 的数据下载链接"""
    r = requests.get(CMR_GRANULE, params={"concept_id": granule_id},
                     timeout=90, headers={"Accept": "application/json"})
    r.raise_for_status()
    entries = r.json().get("feed", {}).get("entry", [])
    if not entries:
        raise RuntimeError("CMR 无该 granule")
    links = entries[0].get("links", [])
    cands = [l for l in links if l.get("rel", "").endswith("/data#")]
    if not cands:
        cands = [l for l in links if "download" in (l.get("title") or "").lower()]
    if not cands:
        raise RuntimeError("没有数据链接，共 %d 个链接" % len(links))
    ln = cands[0]
    return ln["href"], float(ln.get("length") or 0) / 1e6


def download(url, dest, retries=3):
    for attempt in range(1, retries + 1):
        s = requests.Session()
        s.trust_env = False
        auth = (USER, PWD)
        cur = url
        try:
            for _ in range(8):
                r = s.get(cur, allow_redirects=False, stream=True,
                          timeout=TIMEOUT,
                          auth=auth if "earthdata.nasa.gov" in cur else None)
                if r.status_code in (401, 403):
                    r.close()
                    raise RuntimeError("HTTP %d（认证失败）" % r.status_code)
                if r.status_code in (301, 302, 303, 307, 308):
                    nxt = r.headers.get("Location")
                    r.close()
                    if not nxt:
                        raise RuntimeError("跳转无 Location")
                    cur = nxt
                    continue
                if r.status_code != 200:
                    raise RuntimeError("HTTP %d" % r.status_code)

                tmp = dest + ".part"
                got = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        got += len(chunk)
                r.close()
                if got < 1024:
                    raise RuntimeError("内容过小 %d 字节" % got)
                os.replace(tmp, dest)
                return got
            raise RuntimeError("跳转次数过多")
        except Exception:
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    return 0


def worker(records):
    for rec in records:
        title = rec["title"]
        dest = os.path.join(OUTDIR, title)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            with _lock:
                _done["ok"] += 1
            continue
        try:
            url, size = get_data_url(rec["id"])
            got = download(url, dest)
            with _lock:
                _done["ok"] += 1
                _done["bytes"] += got
                n = _done["ok"] + _done["fail"]
            print("  [%2d/%2d] v %s  %.1f MB"
                  % (n, len(records), title[:46], got / 1e6), flush=True)
        except Exception as e:
            with _lock:
                _done["fail"] += 1
                _failed.append((title, str(e)[:70]))
            print("  x %s  %s" % (title[:40], str(e)[:60]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    recs = json.load(open(MANIFEST, encoding="utf-8"))
    print("清单 %d 条" % len(recs))

    if args.probe:
        print("\n=== 探测链接格式 ===")
        for r in recs[:3]:
            try:
                url, size = get_data_url(r["id"])
                print("  %s" % r["title"][:44])
                print("     %.1f MB" % size)
                print("     %s" % url[:110])
            except Exception as e:
                print("  x %s  %s" % (r["title"][:40], str(e)[:70]))
        return

    t0 = time.time()
    worker(recs)
    dt = time.time() - t0

    print("\n=== 完成 ===")
    print("成功 %d  失败 %d  用时 %.1f 分钟  %.2f MB"
          % (_done["ok"], _done["fail"], dt / 60, _done["bytes"] / 1e6))
    if _failed:
        for n, e in _failed:
            print("  ", n, "|", e)
    print("输出:", OUTDIR)


if __name__ == "__main__":
    main()
