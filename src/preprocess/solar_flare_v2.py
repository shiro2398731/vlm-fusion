# -*- coding: utf-8 -*-
r"""
太阳耀斑（源 12 空间天气）v2 —— 按通量判据，全部踩 GOES 标准

判据（全部来自 GOES XRS-B 1-8Å 通量，W/m²）：
  X >= 1e-4   → x_flare
  M >= 1e-5   → m_flare
  C >= 1e-6   → c_flare
  否则         → quiet     ← B 级以下 = 背景水平（NOAA 定义）

数据：
  事件表：HEK 全年 34,867 条（已存 hek_flares_2024_daily.json）
  通量  ：NOAA GOES-18 EXIS-L1b-SFXR（AWS 公开桶，每 5 分钟一个 nc）

流程：
  1) 读事件表 → 每条取峰值时刻
  2) 找对应 SFXR 文件 → 读 irradiance_xrsb1 → 按通量定级（覆盖 HEK 的等级）
  3) 取 SDO AIA 193Å 同时在的帧 → 双产品渲染

用法:
  python solar_flare_v2.py --max 50      # 试跑
  python solar_flare_v2.py               # 全量
"""
import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
import requests
import numpy as np
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
S = requests.Session(); S.trust_env = False

RAW = r"C:\vlmdisaster\data\raw\12_space_weather"
SDO_DIR = os.path.join(RAW, "sdo")
SFXR_DIR = os.path.join(RAW, "sfxr")
OUT = r"C:\vlmdisaster\datasets\12_space_weather\solar_flare"
IMG = os.path.join(OUT, "images")
HEK = os.path.join(RAW, "hek_flares_2024_daily.json")
SDO_BASE = "https://sdo.gsfc.nasa.gov/assets/img/browse/%s/"

def s3_bucket(year):
    """按年份选 GOES 卫星桶（SFXR 覆盖：16=2021-2025, 18=2022-2026）"""
    return "noaa-goes16" if year <= 2021 else "noaa-goes18"

S3 = "https://noaa-goes18.s3.amazonaws.com/EXIS-L1b-SFXR"
RES, BAND, SIZE = "1024", "0193", 384

TH = {"X": 1e-4, "M": 1e-5, "C": 1e-6}


def flux_grade(f):
    if f is None or not np.isfinite(f):
        return None
    if f >= TH["X"]: return "x_flare"
    if f >= TH["M"]: return "m_flare"
    if f >= TH["C"]: return "c_flare"
    return "quiet"


def doy(d):
    return d.strftime("%j")          # 三位，如 182  不能写成 "%Y%j"


def fetch_flux(when):
    """取 when 时刻的 XRS-B 通量（W/m²）"""
    d = when.date()
    hh = when.hour
    base = "%s/%s/%s/%02d" % (S3, d.year, doy(d), hh)
    # 列目录拿文件名
    try:
        r = S.get("https://%s.s3.amazonaws.com/?list-type=2&prefix=EXIS-L1b-SFXR/%s/%s/%02d/&max-keys=200"
                  % (s3_bucket(d.year), d.year, doy(d), hh), timeout=60)
        keys = re.findall(r"<Key>([^<]+)</Key>", r.text)
    except Exception:
        return None
    if not keys:
        return None
    # 按起始秒挑最接近的
    best, bd = None, None
    for k in keys:
        m = re.search(r"_s(\d+)_e", k)      # YYYYDDDHHMMSS(0-9 位小数秒)
        if not m:
            continue
        s = m.group(1)
        try:
            t = datetime(int(s[0:4]), 1, 1) + timedelta(
                days=int(s[4:7]) - 1,
                hours=int(s[7:9]), minutes=int(s[9:11]), seconds=int(s[11:13]))
            dlt = abs((t - when).total_seconds())
        except Exception:
            continue
        if bd is None or dlt < bd:
            best, bd = k, dlt
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
        t = np.asarray(ds.variables["time"][:]).astype("float64")
        ds.close()
        arr[arr <= 0] = np.nan
        if not np.isfinite(arr).any():
            return None
        # 取该文件内的最大值（该 5 分钟窗口的峰值）
        return float(np.nanmax(arr))
    except Exception:
        return None


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
        d = ImageDraw.Draw(im)
        cx, cy = mark
        x, y = SIZE / 2 + cx / 1000.0 * (SIZE / 2), SIZE / 2 - cy / 1000.0 * (SIZE / 2)
        if 0 <= x < SIZE and 0 <= y < SIZE:
            d.ellipse([x - 18, y - 18, x + 18, y + 18], outline=(255, 80, 40), width=2)
    im.save(dst)


ANS = {"x_flare": "X-class solar flare: XRS-B peak flux at or above 1e-4 W/m2.",
       "m_flare": "M-class solar flare: XRS-B peak flux 1e-5 to 1e-4 W/m2.",
       "c_flare": "C-class solar flare: XRS-B peak flux 1e-6 to 1e-5 W/m2.",
       "quiet": "Quiet solar conditions: XRS-B flux below 1e-6 W/m2 (B-class background)."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--quiet-n", type=int, default=400, help="额外挑多少个 quiet 帧")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for f in os.listdir(IMG):
        if f.endswith(".png"):
            os.remove(os.path.join(IMG, f))
    os.makedirs(SDO_DIR, exist_ok=True)

    rows = json.load(open(HEK, encoding="utf-8"))
    seen, ev = set(), []
    for r in rows:
        k = (r.get("event_peaktime"), r.get("fl_goescls"))
        if k in seen:
            continue
        seen.add(k)
        g = r.get("event_peaktime")
        if g:
            ev.append(r)
    ev.sort(key=lambda r: r["event_peaktime"])
    print("事件(去重):", len(ev))
    if args.max:
        ev = ev[:args.max]

    cache, recs, cnt = {}, [], Counter()
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

        flux = fetch_flux(peak)
        lab = flux_grade(flux)
        if lab is None or lab == "quiet":
            continue      # quiet 单独抽

        stamp = peak.strftime("%Y%m%d_%H%M%S")
        raw = os.path.join(SDO_DIR, "%s.jpg" % stamp)
        if not download_img(url, raw):
            continue
        hpc = (r.get("hpc_x") or 0, r.get("hpc_y") or 0)
        for sfx, mk, prod in (("", None, "intrinsic_aia193"),
                              ("_loc", hpc, "feature_flare_location")):
            png = os.path.join(IMG, "%s_%s%s.png" % (stamp, lab, sfx))
            render(raw, png, mk)
            recs.append({"conversations": [
                {"from": "human", "value": "<image>This is an SDO AIA 193A EUV image of the Sun. "
                                            "Classify solar activity: quiet, c_flare, m_flare, or x_flare."},
                {"from": "gpt", "value": ANS[lab]}],
                "images": [png], "product": prod, "label": lab,
                "xrsb_flux": flux, "hek_class": r.get("fl_goescls"),
                "peak": peak.isoformat()})
        cnt[lab] += 1
        if i % 50 == 0:
            print("  %d/%d %s" % (i, len(ev), dict(cnt)), flush=True)

    # 抽 quiet：随机日子、随机时刻，通量 < 1e-6
    print("\n抽 quiet 帧...")
    import random
    random.seed(2024)
    days = sorted(cache.keys())
    got = 0
    tries = 0
    while got < args.quiet_n and tries < args.quiet_n * 6 and days:
        tries += 1
        d = random.choice(days)
        frames = cache[d]
        if not frames:
            continue
        hh, mm, ss = random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)
        when = datetime(d.year, d.month, d.day, hh, mm, ss)
        url, dl = nearest(frames, when)
        if url is None or dl > 600:
            continue
        flux = fetch_flux(when)
        lab = flux_grade(flux)
        if lab != "quiet":
            continue
        stamp = when.strftime("%Y%m%d_%H%M%S")
        raw = os.path.join(SDO_DIR, "%s.jpg" % stamp)
        if not download_img(url, raw):
            continue
        for sfx, prod in (("", "intrinsic_aia193"), ("_loc", "feature_flare_location")):
            png = os.path.join(IMG, "%s_quiet%s.png" % (stamp, sfx))
            render(raw, png)
            recs.append({"conversations": [
                {"from": "human", "value": "<image>This is an SDO AIA 193A EUV image of the Sun. "
                                            "Classify solar activity: quiet, c_flare, m_flare, or x_flare."},
                {"from": "gpt", "value": ANS["quiet"]}],
                "images": [png], "product": prod, "label": "quiet",
                "xrsb_flux": flux, "peak": when.isoformat()})
        cnt["quiet"] += 1
        got += 1
        if got % 50 == 0:
            print("  quiet %d" % got, flush=True)

    # 划分
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
    print("\n事件数:", len(recs) // 2, "| train", len(train), "| val", len(val))
    print("分布:", dict(cnt))
    print("val:", dict(Counter(x["label"] for x in val)))


if __name__ == "__main__":
    main()
