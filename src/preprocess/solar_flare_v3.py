# -*- coding: utf-8 -*-
r"""
太阳耀斑（源 12）v3 —— 多年份，quiet 来自极小时年

判据（GOES XRS-B 1-8Å 通量，W/m²，全部国际标准）：
  X >= 1e-4 / M >= 1e-5 / C >= 1e-6 / 其余 = quiet（B 级背景）

通量来源（两条路）：
  2019-2021 → NCEI GOES-15 XRS 1min CSV（B_AVG，**除以 0.7** 去掉 NOAA 缩放因子）
  2022-2024 → AWS GOES-16/18 EXIS-L1b-SFXR（irradiance_xrsb1，已校正）

分配：
  耀斑样本（C/M/X）→ 2024 年（活动峰，量大）
  quiet 样本      → 2019-2021 年（活动极小，背景 < 1e-6）

用法:
  python solar_flare_v3.py --quiet-years 2019,2020,2021 --quiet-n 300
"""
import os
import re
import sys
import csv
import io
import json
import time
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import requests
import numpy as np
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
S = requests.Session(); S.trust_env = False

RAW = r"C:\vlmdisaster\data\raw\12_space_weather"
SDO_DIR = os.path.join(RAW, "sdo")
SFXR_DIR = os.path.join(RAW, "sfxr")
NCEI_DIR = os.path.join(RAW, "ncei_xrs")
OUT = r"C:\vlmdisaster\datasets\12_space_weather\solar_flare"
IMG = os.path.join(OUT, "images")
HEK2024 = os.path.join(RAW, "hek_flares_2024_daily.json")
HEK2021 = os.path.join(RAW, "hek_flares_2021_daily.json")

SDO_BASE = "https://sdo.gsfc.nasa.gov/assets/img/browse/%s/"
NCEI = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/%d/%02d/goes15/csv/"
RES, BAND, SIZE = "1024", "0193", 384
TH = {"X": 1e-4, "M": 1e-5, "C": 1e-6}
B_SCALE = 0.7          # GOES 8-15 B 通道 NOAA 缩放因子

_cache_month = {}


def flux_grade(f):
    if f is None or not np.isfinite(f):
        return None
    if f >= TH["X"]: return "x_flare"
    if f >= TH["M"]: return "m_flare"
    if f >= TH["C"]: return "c_flare"
    return "quiet"


def ncei_month(year, month):
    """下载并缓存 NCEI 月度 XRS CSV → {datetime: B_AVG}"""
    key = (year, month)
    if key in _cache_month:
        return _cache_month[key]
    local = os.path.join(NCEI_DIR, "g15_xrs_1m_%d%02d.csv" % (year, month))
    txt = None
    if os.path.exists(local):
        txt = open(local, encoding="utf-8", errors="replace").read()
    else:
        try:
            r = S.get(NCEI % (year, month) + "g15_xrs_1m_%d%02d01_%d%02d%02d.csv"
                      % (year, month, year, month, 31 if month in (1,3,5,7,8,10,12)
                         else 29 if month == 2 else 30), timeout=120)
            if r.status_code != 200:
                _cache_month[key] = {}
                return {}
            txt = r.text
            os.makedirs(NCEI_DIR, exist_ok=True)
            open(local, "w", encoding="utf-8").write(txt)
        except Exception:
            _cache_month[key] = {}
            return {}
    d = {}
    hdr = None
    for line in txt.split("\n"):
        if line.startswith("time_tag,"):
            hdr = line.strip().split(",")
            continue
        if hdr is None or not line.strip() or line[0] == ":":
            continue
        parts = line.split(",")
        if len(parts) != len(hdr):
            continue
        try:
            dt = datetime.strptime(parts[0][:19], "%Y-%m-%d %H:%M:%S")
            b = float(parts[hdr.index("B_AVG")])
        except Exception:
            continue
        if b > 0:
            d[dt] = b / B_SCALE
    _cache_month[key] = d
    return d


def s3_bucket(year):
    return "noaa-goes16" if year <= 2021 else "noaa-goes18"


def doy(d):
    return d.strftime("%j")


def sfxr_flux(when):
    d = when.date()
    try:
        r = S.get("https://%s.s3.amazonaws.com/?list-type=2&prefix=EXIS-L1b-SFXR/%s/%s/%02d/&max-keys=200"
                  % (s3_bucket(d.year), d.year, doy(d), when.hour), timeout=60)
        keys = re.findall(r"<Key>([^<]+)</Key>", r.text)
    except Exception:
        return None
    best, bd = None, None
    for k in keys:
        m = re.search(r"_s(\d+)_e", k)
        if not m:
            continue
        s = m.group(1)
        try:
            t = datetime(int(s[0:4]), 1, 1) + timedelta(
                days=int(s[4:7]) - 1, hours=int(s[7:9]),
                minutes=int(s[9:11]), seconds=int(s[11:13]))
            dl = abs((t - when).total_seconds())
        except Exception:
            continue
        if bd is None or dl < bd:
            best, bd = k, dl
    if best is None or bd > 900:
        return None
    local = os.path.join(SFXR_DIR, os.path.basename(best))
    if not os.path.exists(local):
        try:
            rr = S.get("https://%s.s3.amazonaws.com/" % s3_bucket(d.year) + best, timeout=90)
            if rr.status_code != 200:
                return None
            os.makedirs(SFXR_DIR, exist_ok=True)
            open(local, "wb").write(rr.content)
        except Exception:
            return None
    try:
        from netCDF4 import Dataset
        ds = Dataset(local)
        arr = np.asarray(ds.variables["irradiance_xrsb1"][:]).astype("float64")
        ds.close()
        arr[arr <= 0] = np.nan
        return float(np.nanmax(arr)) if np.isfinite(arr).any() else None
    except Exception:
        return None


def flux_at(when):
    """统一入口：2019-2021 走 NCEI，2022+ 走 SFXR"""
    if when.year <= 2021:
        d = ncei_month(when.year, when.month)
        if not d:
            return None
        best, bd = None, None
        for t, v in d.items():
            dl = abs((t - when).total_seconds())
            if dl < 300 and (bd is None or dl < bd):
                best, bd = v, dl
        return best
    return sfxr_flux(when)


def list_day(day):
    u = SDO_BASE % day.strftime("%Y/%m/%d")
    try:
        r = S.get(u, timeout=60)
    except Exception:
        return []
    pat = re.compile(r'href="(\d{8})_(\d{6})_%s_%s\.jpg"' % (RES, BAND))
    return [(m.group(2), u + "%s_%s_%s_%s.jpg" % (m.group(1), m.group(2), RES, BAND))
            for m in pat.finditer(r.text)]


def nearest(frames, when):
    best, bd = None, None
    for hhmmss, url in frames:
        t = datetime(when.year, when.month, when.day,
                     int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6]))
        dl = abs((t - when).total_seconds())
        if bd is None or dl < bd:
            best, bd = url, dl
    return best, bd


def download_img(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        return True
    for _ in range(3):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                open(path, "wb").write(r.content)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def render(src, dst, mark=None):
    im = Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    if mark:
        cx, cy = mark
        x, y = SIZE / 2 + cx / 1000.0 * (SIZE / 2), SIZE / 2 - cy / 1000.0 * (SIZE / 2)
        if 0 <= x < SIZE and 0 <= y < SIZE:
            ImageDraw.Draw(im).ellipse([x - 18, y - 18, x + 18, y + 18],
                                       outline=(255, 80, 40), width=2)
    im.save(dst)


ANS = {"x_flare": "X-class solar flare: XRS-B peak flux at or above 1e-4 W/m2.",
       "m_flare": "M-class solar flare: XRS-B peak flux between 1e-5 and 1e-4 W/m2.",
       "c_flare": "C-class solar flare: XRS-B peak flux between 1e-6 and 1e-5 W/m2.",
       "quiet": "Quiet solar conditions: XRS-B flux below 1e-6 W/m2 (B-class background)."}
Q = ("<image>This is an SDO AIA 193A EUV image of the Sun. "
     "Classify solar activity: quiet, c_flare, m_flare, or x_flare.")


def add(recs, cnt, src, lab, prod, when, flux, extra=None, mark=None):
    stamp = when.strftime("%Y%m%d_%H%M%S")
    raw = os.path.join(SDO_DIR, "%s.jpg" % stamp)
    if not download_img(src, raw):
        return False
    png = os.path.join(IMG, "%s_%s%s.png" % (stamp, lab, "_loc" if mark else ""))
    render(raw, png, mark)
    r = {"conversations": [{"from": "human", "value": Q},
                           {"from": "gpt", "value": ANS[lab]}],
         "images": [png], "product": prod, "label": lab,
         "xrsb_flux": flux, "peak": when.isoformat()}
    if extra:
        r.update(extra)
    recs.append(r)
    cnt[lab] += 1
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=0, help="耀斑事件上限（0=按类别配额）")
    ap.add_argument("--quiet-n", type=int, default=300)
    ap.add_argument("--quiet-years", default="2019,2021")
    ap.add_argument("--quota-x", type=int, default=0, help="X 级取多少（0=全取）")
    ap.add_argument("--quota-m", type=int, default=300)
    ap.add_argument("--quota-c", type=int, default=600)
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for f in os.listdir(IMG):
        if f.endswith(".png"):
            os.remove(os.path.join(IMG, f))
    os.makedirs(SDO_DIR, exist_ok=True)

    recs, cnt, cache = [], Counter(), {}

    # ---------- 1) 耀斑（2024） ----------
    rows = json.load(open(HEK2024, encoding="utf-8"))
    seen, ev = set(), []
    for r in rows:
        k = (r.get("event_peaktime"), r.get("fl_goescls"))
        if k in seen or not r.get("event_peaktime"):
            continue
        seen.add(k)
        ev.append(r)
    ev.sort(key=lambda r: r["event_peaktime"])
    print("2024 事件:", len(ev))

    # 按 HEK 等级分层抽样（避免全跑 3 万条）
    if not args.events:
        import random as _rnd
        _rnd.seed(420)
        buckets = defaultdict(list)
        for r in ev:
            c = (r.get("fl_goescls") or "?")
            buckets[c[0] if c else "?"].append(r)
        picked = []
        for lv, q in (("X", args.quota_x), ("M", args.quota_m), ("C", args.quota_c)):
            lst = buckets.get(lv, [])
            if q and len(lst) > q:
                picked += _rnd.sample(lst, q)
            else:
                picked += lst
            print("  %s 级: 总 %d -> 取 %d" % (lv, len(lst), min(len(lst), q) if q else len(lst)))
        picked.sort(key=lambda r: r["event_peaktime"])
        ev = picked
    elif args.events:
        ev = ev[:args.events]

    for i, r in enumerate(ev, 1):
        try:
            peak = datetime.fromisoformat(r["event_peaktime"])
        except Exception:
            continue
        day = peak.date()
        if day not in cache:
            cache[day] = list_day(day)
        frames = cache[day]
        if not frames:
            continue
        url, dl = nearest(frames, peak)
        if url is None or dl > 1800:
            continue
        flux = flux_at(peak)
        lab = flux_grade(flux)
        if lab is None or lab == "quiet":
            continue
        hpc = (r.get("hpc_x") or 0, r.get("hpc_y") or 0)
        add(recs, cnt, url, lab, "intrinsic_aia193", peak, flux, {"hek_class": r.get("fl_goescls")})
        add(recs, cnt, url, lab, "feature_flare_location", peak, flux,
            {"hek_class": r.get("fl_goescls")}, mark=hpc)
        if i % 50 == 0:
            print("  事件 %d/%d %s" % (i, len(ev), dict(cnt)), flush=True)

    # ---------- 2) quiet（2019-2021 极小时） ----------
    print("\n抽 quiet（%s）..." % args.quiet_years)
    import random
    random.seed(2019)
    years = [int(y) for y in args.quiet_years.split(",")]
    got, tries = 0, 0
    while got < args.quiet_n and tries < args.quiet_n * 30:
        tries += 1
        y = random.choice(years)
        m = random.randint(1, 12)
        dd = random.randint(1, 28)
        h, mi = random.randint(0, 23), random.randint(0, 59)
        when = datetime(y, m, dd, h, mi)
        day = when.date()
        if day not in cache:
            cache[day] = list_day(day)
        frames = cache[day]
        if not frames:
            continue
        url, dl = nearest(frames, when)
        if url is None or dl > 600:
            continue
        flux = flux_at(when)
        if flux_grade(flux) != "quiet":
            continue
        if add(recs, cnt, url, "quiet", "intrinsic_aia193", when, flux):
            got += 1
            if got % 25 == 0:
                print("  quiet %d/%d (试了 %d 次)" % (got, args.quiet_n, tries), flush=True)

    # ---------- 3) 划分 ----------
    by = defaultdict(list)
    for x in recs:
        by[x["label"]].append(x)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(30, max(2, len(lst) // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_flare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_flare_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n总图 %d | train %d | val %d" % (len(recs), len(train), len(val)))
    print("分布:", dict(cnt))
    print("val:", dict(Counter(x["label"] for x in val)))


if __name__ == "__main__":
    main()
