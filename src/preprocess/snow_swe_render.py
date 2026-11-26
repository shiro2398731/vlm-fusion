# -*- coding: utf-8 -*-
r"""
积雪 × 微波（AMSR2 日积雪当量 SWE）切窗 + 结构化 JSON（新元素：snow_swe）

物理链路
--------
  AMSR2 被动微波亮温 → 积雪辐射传输模型 → 雪水当量 SWE（mm）
  SWE 高 = 积雪厚/雪灾风险
真值/判据
---------
  判据：窗内平均 SWE ≥ 15 mm 视为有积雪（本文设定，对应约 5cm 新雪量级）
任务
----
  detected：窗内平均 SWE 是否 ≥ 15 mm
  数值字段：mean_swe_mm（只能从图上读）
  severity：none <15 / light 15–30 / moderate 30–60 / heavy ≥60

用法
----
  .\venv\Scripts\python.exe src\preprocess\snow_swe_render.py
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
import h5py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "swe_amsr2")
OUT = os.path.join(ROOT, "datasets", "04_passive_microwave", "snow_swe")
IMG = os.path.join(OUT, "images")

WIN = 16
STRIDE = 16
MIN_VALID = 0.4
POS_PER_DAY = 8
NEG_PER_DAY = 8
SEED = 42
OUT_SIZE = 384
LATB = (25.0, 55.0)
LONB = (75.0, 135.0)

PROMPT = ("This is an AMSR2 snow water equivalent (SWE) map in mm over northern China. High SWE "
          "means deep snow cover. Output a JSON object with fields: disaster_type, detected "
          "(true/false), source, physical_quantity, severity, mean_swe_mm, criterion. "
          "severity must be one of these exact words: none / light / moderate / heavy.")


def severity_of(v):
    if v < 15:
        return "none"
    if v < 30:
        return "light"
    if v < 60:
        return "moderate"
    return "heavy"


def load_swe(fp):
    with h5py.File(fp, "r") as h:
        g = h["HDFEOS/GRIDS/Northern Hemisphere"]
        swe = g["Data Fields/SWE_NorthernDaily"][:].astype("float32")
        lat = g["lat"][:].astype("float64")
        lon = g["lon"][:].astype("float64")
    swe[(swe >= 250) | (swe < 0)] = np.nan
    lat[(lat < -90) | (lat > 90)] = np.nan
    lon[(lon < -180) | (lon > 180)] = np.nan
    return swe, lat, lon


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="Blues", vmin=0, vmax=120, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("SWE (mm)", fontsize=7)
    fig.text(0.04, 0.075, "AMSR2 daily SWE, 25 km", fontsize=6)
    fig.text(0.04, 0.035, "white = no retrieval", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(RAW, "*.he5")))
    print("SWE 日文件 %d 个" % len(files))
    if a.step > 1:
        files = files[::a.step]
    if a.limit_days:
        files = files[:a.limit_days]

    records = []
    random.seed(SEED)
    for fi, fp in enumerate(files, 1):
        m = re.search(r"(\d{8})", os.path.basename(fp))
        stamp = m.group(1) if m else "unknown"
        swe, lat, lon = load_swe(fp)
        h, w = swe.shape
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                ws = swe[r0:r0 + WIN, c0:c0 + WIN]
                if np.isfinite(ws).mean() < MIN_VALID:
                    continue
                la = float(np.nanmean(lat[r0:r0 + WIN, c0:c0 + WIN]))
                lo = float(np.nanmean(lon[r0:r0 + WIN, c0:c0 + WIN]))
                if not (LATB[0] < la < LATB[1] and LONB[0] < lo < LONB[1]):
                    continue
                ms = float(np.nanmean(ws))
                item = (r0, c0, la, lo, ms)
                if ms >= 15:
                    pos.append(item)
                elif ms < 3:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for det, (r0, c0, la, lo, ms) in [(True, x) for x in pos[:POS_PER_DAY]] + \
                [(False, x) for x in neg[:NEG_PER_DAY]]:
            sev = severity_of(ms)
            nm = "swe_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            render(swe[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "snow_cover",
                "detected": bool(det),
                "source": "MW",
                "physical_quantity": "swe_mm",
                "severity": sev,
                "mean_swe_mm": round(ms, 1),
                "criterion": ("mean SWE >= 15 mm indicates snow cover (study-defined); "
                              "severity bands 15/30/60 mm"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp, "mean_swe_mm": round(ms, 1),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if fi % 10 == 0 or fi == len(files):
            print("  [%d/%d] %s 累计 %d" % (fi, len(files), stamp, len(records)))

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

    json.dump(train, open(os.path.join(OUT, "train_snow_swe_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_snow_swe_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
