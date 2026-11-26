# -*- coding: utf-8 -*-
r"""
叶绿素 a 切窗 + 结构化 JSON（新元素：bloom_chla）

物理链路
--------
  VIIRS/OLCI 海洋水色 → 离水辐亮度 → 叶绿素 a 浓度（mg/m³）
  高叶绿素浓度 = 浮游植物大量繁殖（水华/赤潮/富营养化）
真值/判据
---------
  判据：窗内 P90 叶绿素 a >= 5 mg/m³ 视为水华（文献常用富营养化/藻华阈值，本文采用）
  与 green_tide（Sentinel-2 漂浮藻类指数）不同源、不同物理量

任务设计（防塌缩）
------------------
  detected：窗内 P90 叶绿素 a 是否 >= 5（读数 + 阈值）
  数值字段：p90_chla（mg/m³）
  severity：none<1 / light 1-5 / moderate 5-20 / heavy>=20（本文设定）

用法
----
  .\venv\Scripts\python.exe src\preprocess\chla_bloom_render_json.py --top-days 5
  .\venv\Scripts\python.exe src\preprocess\chla_bloom_render_json.py --top-days 120
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
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "01_optical_vnir", "chla_modis")
OUT = os.path.join(ROOT, "datasets", "01_optical_vnir", "bloom_chla_modis")
IMG = os.path.join(OUT, "images")

WIN = 64
STRIDE = 64
MIN_VALID = 0.5
POS_PER_DAY = 4
NEG_PER_DAY = 4
BLOOM_MIN = 5.0        # mg/m³
SEED = 42
OUT_SIZE = 384

PROMPT = ("This is a satellite ocean-colour chlorophyll-a map (mg/m^3) over a coastal sea. "
          "High chlorophyll indicates phytoplankton blooms / eutrophication. Output a JSON "
          "object with fields: disaster_type, detected (true/false), source, "
          "physical_quantity, severity, p90_chla, criterion. severity must be one of these "
          "exact words: none / light / moderate / heavy.")


def severity_of(v):
    if v < 1:
        return "none"
    if v < 5:
        return "light"
    if v < 20:
        return "moderate"
    return "heavy"


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="YlGn", norm=LogNorm(vmin=0.01, vmax=100),
                   aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("chlorophyll a (mg/m3)", fontsize=6.5)
    fig.text(0.04, 0.075, "NOAA ERDDAP nesdisNPPN20S3ASCIDINEOFDaily, 9 km", fontsize=6)
    fig.text(0.04, 0.035, "white = no data", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def p90(v):
    return float(np.percentile(v, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-days", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(RAW, "chla_*.nc")))
    print("叶绿素月块 %d 个" % len(files))
    if not files:
        return

    # 第一遍：逐月算每天的最佳窗 P90，挑强水华日
    day_best = {}
    meta = {}
    for f in files:
        ds = Dataset(f)
        chl = np.ma.filled(np.array(ds.variables["chlorophyll"][:], dtype="float32"), -999.0)
        lat = np.array(ds.variables["latitude"][:], dtype="float64")
        lon = np.array(ds.variables["longitude"][:], dtype="float64")
        t = ds.variables["time"]
        times = t[:]
        import datetime
        for i in range(chl.shape[0]):
            stamp = str(datetime.datetime.utcfromtimestamp(float(times[i])))[:10].replace("-", "")
            arr = chl[i]
            if arr.ndim > 2:
                arr = arr[0]
            arr = np.where((arr < 0) | (arr > 200), np.nan, arr)
            h, w = arr.shape
            best = 0.0
            for r0 in range(0, h - WIN + 1, STRIDE):
                for c0 in range(0, w - WIN + 1, STRIDE):
                    ww = arr[r0:r0 + WIN, c0:c0 + WIN]
                    if np.isfinite(ww).mean() < MIN_VALID:
                        continue
                    v = float(np.nanpercentile(ww, 90))
                    if v > best:
                        best = v
            day_best[stamp] = best
            meta[stamp] = f
    days = sorted([d for d, v in day_best.items() if v >= BLOOM_MIN], key=lambda d: -day_best[d])
    if args.top_days:
        days = days[:args.top_days]
    print("强水华日 %d 个（取 %d）" % (len([d for d, v in day_best.items() if v >= BLOOM_MIN]), len(days)))
    if not days:
        print("没有可用水华日")
        return

    # 第二遍：逐日切窗渲染
    by_file = collections.defaultdict(list)
    for d in days:
        by_file[meta[d]].append(d)
    records = []
    random.seed(SEED)
    done = 0
    for f, ds_days in by_file.items():
        ds = Dataset(f)
        chl = np.ma.filled(np.array(ds.variables["chlorophyll"][:], dtype="float32"), -999.0)
        lat = np.array(ds.variables["latitude"][:], dtype="float64")
        lon = np.array(ds.variables["longitude"][:], dtype="float64")
        times = ds.variables["time"][:]
        ds.close()
        import datetime
        for i in range(chl.shape[0]):
            stamp = str(datetime.datetime.utcfromtimestamp(float(times[i])))[:10].replace("-", "")
            if stamp not in ds_days:
                continue
            arr = chl[i]
            if arr.ndim > 2:
                arr = arr[0]
            arr = np.where((arr < 0) | (arr > 200), np.nan, arr)
            h, w = arr.shape
            pos, neg = [], []
            for r0 in range(0, h - WIN + 1, STRIDE):
                for c0 in range(0, w - WIN + 1, STRIDE):
                    ww = arr[r0:r0 + WIN, c0:c0 + WIN]
                    if np.isfinite(ww).mean() < MIN_VALID:
                        continue
                    v90 = float(np.nanpercentile(ww, 90))
                    la = float(np.nanmean(lat[r0:r0 + WIN]))
                    lo = float(np.nanmean(lon[c0:c0 + WIN]))
                    item = (r0, c0, la, lo, v90)
                    if v90 >= BLOOM_MIN:
                        pos.append(item)
                    elif v90 < 1.0:
                        neg.append(item)
            random.shuffle(pos)
            random.shuffle(neg)
            for (r0, c0, la, lo, v90) in pos[:POS_PER_DAY] + neg[:NEG_PER_DAY]:
                sev = severity_of(v90)
                det = v90 >= BLOOM_MIN
                name = "bloommodis_%s_r%d_c%d.png" % (stamp, r0, c0)
                p = os.path.join(IMG, name)
                render(arr[r0:r0 + WIN, c0:c0 + WIN], p, stamp, la, lo)
                ans = {
                    "disaster_type": "algal_bloom",
                    "detected": bool(det),
                    "source": "OPT",
                    "physical_quantity": "chlorophyll_a",
                    "severity": sev,
                    "p90_chla": round(v90, 2),
                    "criterion": ("window P90 chlorophyll-a >= 5 mg/m3 indicates a bloom "
                                  "(study-defined threshold); severity bands study-defined"),
                }
                records.append({
                    "image": p, "label": sev, "stamp": stamp, "p90_chla": round(v90, 2),
                    "conversations": [
                        {"from": "human", "value": "<image>" + PROMPT},
                        {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                    ],
                })
            done += 1
            if done % 10 == 0:
                print("  %d/%d 天  累计 %d" % (done, len(days), len(records)))

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

    json.dump(train, open(os.path.join(OUT, "train_bloom_chla_modis_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_bloom_chla_modis_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
