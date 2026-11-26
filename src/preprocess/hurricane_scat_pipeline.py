# -*- coding: utf-8 -*-
r"""
台风 × ASCAT 微波风场 切窗管线（新元素：hurricane_scat）

物理链路
--------
  Metop ASCAT 散射计 → 海面 10m 风矢量（x/y 分量，m/s）→ 风速
真值
----
  IBTrACS 台风最佳路径：每日台风中心位置（2020–2024）
任务
----
  detected：窗口是否位于台风中心 8° 内（真值）
  数值字段：max_wind_ms（窗内最大风速，m/s；只能从图上读）
  severity：<17.2 none / 17.2–24.5 light / 24.5–32.7 moderate / >=32.7 heavy（蒲福 8/10/12 级）

用法
----
  .\venv\Scripts\python.exe src\preprocess\hurricane_scat_pipeline.py --limit-days 3
  .\venv\Scripts\python.exe src\preprocess\hurricane_scat_pipeline.py --limit-days 120
"""
import argparse
import collections
import csv
import datetime
import glob
import json
import math
import os
import random
import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
IBT = os.path.join(ROOT, "data", "raw", "13_ibtracs", "ibtracs.csv")
ASCAT = os.path.join(ROOT, "data", "raw", "07_scatterometer", "ascat_day")
OUT = os.path.join(ROOT, "datasets", "07_scatterometer", "hurricane_scat")
IMG = os.path.join(OUT, "images")

URL = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdQMwind1day_LonPM180.nc"
       "?x_wind[(%s):1:(%s)][(10.0)][(%s):1:(%s)][(%s):1:(%s)]"
       ",y_wind[(%s):1:(%s)][(10.0)][(%s):1:(%s)][(%s):1:(%s)]")
BOX = 35.0  # 台风周围 ±35°
WIN = 64
STRIDE = 32
MIN_VALID = 0.3
POS_PER_DAY = 2
NEG_PER_DAY = 3
SEED = 42
OUT_SIZE = 384
GALE = 17.2

PROMPT = ("This is an ASCAT scatterometer ocean wind-speed map (m/s). A tropical cyclone "
          "shows a compact high-wind core. Output a JSON object with fields: "
          "disaster_type, detected (true/false), source, physical_quantity, severity, "
          "max_wind_ms, criterion. severity must be one of these exact words: "
          "none / light / moderate / heavy.")


def severity_of(v):
    if v < 17.2:
        return "none"
    if v < 24.5:
        return "light"
    if v < 32.7:
        return "moderate"
    return "heavy"


def parse_ibtracs():
    """返回 [(date, sid, name, lat, lon, wind_kt)]，2020–2024、|lat|<=55"""
    out = []
    with open(IBT, encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        next(r)  # units
        for row in r:
            if len(row) < 11:
                continue
            try:
                season = int(float(row[1]))
            except ValueError:
                continue
            if not (2020 <= season <= 2022):
                continue
            nature = row[7].strip()
            if nature not in ("TS", "SS", "ET", ""):
                continue
            try:
                lat, lon = float(row[8]), float(row[9])
                wind = float(row[10]) if row[10] else 0.0
            except ValueError:
                continue
            if abs(lat) > 55:
                continue
            d = row[6].strip()[:10].replace("-", "")
            if len(d) != 8:
                continue
            out.append((d, row[0].strip(), row[5].strip(), lat, lon, wind))
    # 每天每台风取最大风速
    best = {}
    for (d, sid, name, lat, lon, wind) in out:
        k = (d, sid)
        if k not in best or wind > best[k][3]:
            best[k] = (lat, lon, wind, wind)
    res = [(d, sid, name, lat, lon, wind) for (d, sid), (lat, lon, wind, _) in best.items()]
    return res


def download_day(stamp, lat, lon):
    lat0 = max(-75.0, lat - BOX)
    lat1 = min(75.0, lat + BOX)
    lon0 = max(-180.0, lon - BOX)
    lon1 = min(179.75, lon + BOX)
    dst = os.path.join(ASCAT, "ascat_%s_%d_%d.nc" % (stamp, int(lat), int(lon)))
    if os.path.exists(dst) and os.path.getsize(dst) > 50_000:
        return dst
    t = "%s-%s-%s" % (stamp[:4], stamp[4:6], stamp[6:])
    url = URL % (t, t, lat0, lat1, lon0, lon1, t, t, lat0, lat1, lon0, lon1)
    r = subprocess.run(["curl.exe", "-g", "-L", "--noproxy", "*", "--ssl-no-revoke",
                        "--max-time", "300", "-s", "-o", dst + ".part", url],
                       capture_output=True)
    if os.path.exists(dst + ".part") and os.path.getsize(dst + ".part") > 50_000:
        os.replace(dst + ".part", dst)
        return dst
    return None


def load_day(fp):
    ds = Dataset(fp)
    x = np.ma.filled(np.array(ds.variables["x_wind"][:], dtype="float32"), np.nan)
    y = np.ma.filled(np.array(ds.variables["y_wind"][:], dtype="float32"), np.nan)
    lat = np.array(ds.variables["latitude"][:], dtype="float64")
    lon = np.array(ds.variables["longitude"][:], dtype="float64")
    ds.close()
    x = np.where(np.abs(x) > 100, np.nan, x)
    y = np.where(np.abs(y) > 100, np.nan, y)
    spd = np.sqrt(x * x + y * y)
    spd = np.squeeze(spd)
    if spd.ndim > 2:
        spd = spd[0]
    return spd, lat, lon


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="turbo", vmin=0, vmax=35, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("wind speed (m/s)", fontsize=7)
    fig.text(0.04, 0.075, "Metop ASCAT, daily 0.25 deg", fontsize=6)
    fig.text(0.04, 0.035, "white = no swath coverage", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=120)
    a = ap.parse_args()

    os.makedirs(ASCAT, exist_ok=True)
    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    ev = parse_ibtracs()
    print("IBTrACS 台风日记录 %d 条" % len(ev))
    ev.sort(key=lambda t: -t[5])
    if a.limit_days:
        ev = ev[:a.limit_days]
    print("取强台风日 %d 个" % len(ev))

    records = []
    random.seed(SEED)
    used = 0
    for i, (d, sid, name, lat, lon, wind_kt) in enumerate(ev, 1):
        fp = download_day(d, lat, lon)
        if not fp:
            continue
        spd, latA, lonA = load_day(fp)
        h, w = spd.shape
        # 网格：lat -75..75 (601)，lon -180..179.75 (1440)
        ri = int(np.clip(int(np.argmin(np.abs(latA - lat))), 0, h - WIN))
        ci = int(np.clip(int(np.argmin(np.abs(lonA - lon))), 0, w - WIN))
        win = spd[ri:ri + WIN, ci:ci + WIN]
        if np.isfinite(win).mean() < MIN_VALID:
            continue
        # 台风中心附近 5x5 个候选窗，找有星下点覆盖的
        pos = []
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                rr = int(np.clip(ri + dr * WIN // 2, 0, h - WIN))
                cc = int(np.clip(ci + dc * WIN // 2, 0, w - WIN))
                ww = spd[rr:rr + WIN, cc:cc + WIN]
                if np.isfinite(ww).mean() < MIN_VALID:
                    continue
                mx = float(np.nanmax(ww))
                if mx < 15:
                    continue
                pos.append((rr, cc, float(latA[min(rr + WIN // 2, h - 1)]),
                            float(lonA[min(cc + WIN // 2, w - 1)]), mx))
        pos.sort(key=lambda t: -t[4])
        pos = pos[:POS_PER_DAY]
        # 负样本：远离台风中心 >20°，风速 < 15
        neg = []
        for _ in range(300):
            if len(neg) >= NEG_PER_DAY:
                break
            rr = random.randrange(0, h - WIN + 1)
            cc = random.randrange(0, w - WIN + 1)
            la = float(latA[min(rr + WIN // 2, h - 1)])
            lo = float(lonA[min(cc + WIN // 2, w - 1)])
            if math.hypot(la - lat, (lo - lon) * math.cos(math.radians(lat))) < 20:
                continue
            ww = spd[rr:rr + WIN, cc:cc + WIN]
            if np.isfinite(ww).mean() < MIN_VALID:
                continue
            mx = float(np.nanmax(ww))
            if mx >= 15:
                continue
            neg.append((rr, cc, la, lo, mx))
        picks = [(True,) + p for p in pos[:POS_PER_DAY]] + [(False,) + p for p in neg]
        for det, rr, cc, la, lo, mx in picks:
            sev = severity_of(mx)
            nm = "hscat_%s_r%d_c%d.png" % (d, rr, cc)
            pp = os.path.join(IMG, nm)
            render(spd[rr:rr + WIN, cc:cc + WIN], pp, d, la, lo)
            ans = {
                "disaster_type": "tropical_cyclone",
                "detected": bool(det),
                "source": "SCAT",
                "physical_quantity": "wind_speed",
                "severity": sev,
                "max_wind_ms": round(mx, 1),
                "criterion": ("storm center from IBTrACS; max wind in window; "
                              "severity = Beaufort 8/10/12 thresholds (17.2/24.5/32.7 m/s)"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": d, "max_wind_ms": round(mx, 1),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        used += 1
        if i % 20 == 0 or i == len(ev):
            print("  [%d/%d] %s %s 累计 %d" % (i, len(ev), d, name, len(records)))

    # 去重 + 分层划分
    seen = set()
    uniq = []
    for r in records:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        uniq.append(r)
    random.seed(SEED)
    by = collections.defaultdict(list)
    for r in uniq:
        by[r["label"]].append(r)
    val, train = [], []
    for k in sorted(by):
        lst = by[k][:]
        random.shuffle(lst)
        n = min(max(2, int(round(len(lst) * 0.2))), 30, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_hurricane_scat_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_hurricane_scat_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
