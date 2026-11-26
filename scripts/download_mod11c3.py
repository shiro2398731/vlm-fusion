# -*- coding: utf-8 -*-
"""下载 MODIS 月 LST（MOD11C3 v061）2020-2022，最多 36 个月。"""
import json
from config_local import cred as _cred
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\vlmdisaster\scripts")
EARTHDATA_TOKEN = os.environ.get("EARTHDATA_TOKEN", "") or _cred("EARTHDATA_TOKEN")

ROOT = r"C:\vlmdisaster"
OUT = os.path.join(ROOT, "data", "raw", "03_thermal_infrared", "mod11c3")
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"


def main():
    os.makedirs(OUT, exist_ok=True)
    u = (CMR + "?short_name=MOD11C3&temporal=2020-01-01T00:00:00Z,2022-12-31T23:59:59Z&page_size=2000")
    r = subprocess.run(["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke",
                        "--max-time", "120", "-s", u], capture_output=True)
    j = json.loads(r.stdout.decode("utf-8", errors="replace"))
    urls = []
    for it in j.get("items", []):
        for x in it["umm"].get("RelatedUrls", []):
            if x.get("URL", "").endswith(".hdf"):
                urls.append(x["URL"])
    seen, files = set(), []
    for u2 in urls:
        nm = os.path.basename(u2)
        parts = nm.split(".")
        key = parts[1] if len(parts) > 1 else nm
        if key not in seen:
            seen.add(key)
            files.append(u2)
    files = files[:36]
    print("granules", len(urls), "unique months", len(files))

    ok = skip = fail = 0
    for i, url in enumerate(files, 1):
        name = os.path.basename(url)
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
            skip += 1
            continue
        part = dst + ".part"
        subprocess.run(["curl.exe", "-L", "--noproxy", "*", "--ssl-no-revoke", "--max-time", "600",
                        "-s", "-H", "Authorization: Bearer " + EARTHDATA_TOKEN, "-o", part, url],
                       capture_output=True)
        good = False
        if os.path.exists(part) and os.path.getsize(part) > 1_000_000:
            with open(part, "rb") as f:
                head = f.read(8)
            if head.startswith(b"\x89HDF") or head.startswith(b"CDF") or head.startswith(b"\x0e\x03\x13\x01"):
                os.replace(part, dst)
                good = True
                ok += 1
        if not good:
            fail += 1
            if os.path.exists(part):
                os.remove(part)
        if i % 5 == 0 or i == len(files):
            print("  [%d/%d] ok=%d skip=%d fail=%d" % (i, len(files), ok, skip, fail), flush=True)
    print("完成 ok=%d skip=%d fail=%d -> %s" % (ok, skip, fail, OUT))


if __name__ == "__main__":
    main()
