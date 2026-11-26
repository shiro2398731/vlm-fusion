# -*- coding: utf-8 -*-
r"""
洪水 × 降水（CHIRPS 月降水 + GDACS 洪水事件）切窗 + 结构化 JSON
（新元素：flood_precip）

物理链路
--------
  CHIRPS 月降水 → 月降水距平（相对 2010–2019 同月气候态）→ 强降水/洪涝
真值
----
  GDACS 洪水事件（2020–2022，中国区域）；正样本 = 事件地点附近的窗口
任务
----
  detected：窗口是否在洪水事件 2° 内（真值）
  数值字段：mean_precip_anomaly（mm/月，只能从图上读）
  severity：none<20 / light 20-60 / moderate 60-120 / heavy ≥120（mm）

用法
----
  .\venv\Scripts\python.exe src\preprocess\flood_precip_render.py --limit-events 3
  .\venv\Scripts\python.exe src\preprocess\flood_precip_render.py
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "06_radar_precipitation", "chirps_monthly")
EVENTS = os.path.join(ROOT, "data", "raw", "14_gdacs_floods.json")
OUT = os.path.join(ROOT, "datasets", "06_radar_precipitation", "flood_precip")
IMG = os.path.join(OUT, "images")

WIN = 64
STRIDE = 32
MIN_VALID = 0.5
POS_PER_EVENT = 10
NEG_PER_EVENT = 6
SEED = 42
OUT_SIZE = 384
CLIM0, CLIM1 = 2010, 2019

PROMPT = ("This is a monthly precipitation anomaly map over China (mm per month) versus the "
          "2010-2019 climatology: positive = much wetter than normal. Output a JSON object "
          "with fields: disaster_type, detected (true/false), source, physical_quantity, "
          "severity, mean_precip_anomaly, criterion. severity must be one of these exact "
          "words: none / light / moderate / heavy.")


def severity_of(v):
    if v < 20:
        return "none"
    if v < 60:
        return "light"
    if v < 120:
        return "moderate"
    return "heavy"


def load_events():
    j = json.load(open(EVENTS, encoding="utf-8"))
    ev = []
    for f in j["features"]:
        c = f["geometry"]["coordinates"]
        if not c or len(c) < 2:
            continue
        lo, la = float(c[0]), float(c[1])
        if not (100 < lo < 136 and 15 < la < 46):
            continue
        d = f["properties"].get("fromdate", "")
        try:
            yy, mm = int(d[:4]), int(d[5:7])
        except ValueError:
            continue
        ev.append((yy, mm, la, lo, f["properties"].get("name", "")))
    return ev


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="RdBu", vmin=-60, vmax=180, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("monthly precip anomaly (mm)", fontsize=6.5)
    fig.text(0.04, 0.075, "CHIRPS monthly, 0.1 deg grid", fontsize=6)
    fig.text(0.04, 0.035, "blue = wet anomaly", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-events", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    data, lat, lon = {}, None, None
    for f in sorted(glob.glob(os.path.join(RAW, "chla_*.nc"))):
        m = re.search(r"chla_(\d{4})(\d{2})", os.path.basename(f))
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        ds = Dataset(f)
        p = np.ma.filled(np.array(ds.variables["precip"][:], dtype="float32"), np.nan)
        if lat is None:
            lat = np.array(ds.variables["latitude"][:], dtype="float64")
            lon = np.array(ds.variables["longitude"][:], dtype="float64")
        ds.close()
        data[(y, mo)] = p[0] if p.ndim == 3 else p
    print("降水月份 %d" % len(data))

    ev = load_events()
    print("中国区域 GDACS 洪水事件 %d 个" % len(ev))
    if a.limit_events:
        ev = ev[:a.limit_events]

    records = []
    random.seed(SEED)
    used = 0
    for (y, mo, ela, elo, name) in ev:
        if (y, mo) not in data:
            continue
        clim = [data[(yy, mo)] for yy in range(CLIM0, CLIM1 + 1) if (yy, mo) in data]
        if len(clim) < 5:
            continue
        base = np.nanmean(np.stack(clim), axis=0)
        anom = data[(y, mo)] - base
        h, w = anom.shape
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                wa = anom[r0:r0 + WIN, c0:c0 + WIN]
                if np.isfinite(wa).mean() < MIN_VALID:
                    continue
                la = float(np.nanmean(lat[r0:r0 + WIN]))
                lo = float(np.nanmean(lon[c0:c0 + WIN]))
                dy = (la - ela) * 111.0
                dx = (lo - elo) * 111.0 * np.cos(np.deg2rad(ela))
                dist = float(np.hypot(dy, dx))
                item = (r0, c0, la, lo, float(np.nanmean(wa)), dist)
                if dist <= 220:
                    pos.append(item)
                elif dist > 500 and item[4] < 60:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        stamp = "%04d%02d" % (y, mo)
        for det, (r0, c0, la, lo, mean_a, dist) in [(True, x) for x in pos[:POS_PER_EVENT]] + \
                [(False, x) for x in neg[:NEG_PER_EVENT]]:
            sev = severity_of(mean_a)
            nm = "floodp_%s_%d_r%d_c%d.png" % (stamp, used, r0, c0)
            pp = os.path.join(IMG, nm)
            render(anom[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "flood",
                "detected": bool(det),
                "source": "PRECIP",
                "physical_quantity": "precip_anomaly",
                "severity": sev,
                "mean_precip_anomaly": round(mean_a, 1),
                "criterion": ("flood event location from GDACS (2020-2022); monthly CHIRPS "
                              "anomaly vs 2010-2019 climatology; severity bands study-defined"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "mean_precip_anomaly": round(mean_a, 1),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        used += 1
        print("  [%d/%d] %s %s 正 %d 负 %d 累计 %d" % (used, len(ev), stamp, name, len(pos), len(neg), len(records)))

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

    json.dump(train, open(os.path.join(OUT, "train_flood_precip_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_flood_precip_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
