# -*- coding: utf-8 -*-
r"""
MODIS NDVI 切窗 + 结构化 JSON（新元素：drought_ndvi）

物理链路
--------
  光学 VNIR（MODIS/Terra）→ 地表反射率 → NDVI（归一化植被指数）
  同一月份、多年气候态 → NDVI 距平（植被受水分胁迫的信号）
真值
----
  SPEI-3（标准化降水蒸散指数，3 个月尺度，0.5° 月数据）：
  ≤ -1.0 为中度以上干旱（McKee et al. 1993 的通用分级）
任务
----
  detected：窗口是否处于 SPEI-3 ≤ -1.0 的干旱区（真值）
  数值字段：窗口 NDVI 距平 mean（只能从图上读）
  severity：normal > -0.5 / slight -0.5~-1 / moderate -1~-1.5 / severe ≤ -1.5

注意
----
  SPEIbase v2.9 覆盖 1901–2022，所以本元素先用 2020–2022 共 36 个月；
  补到 v2.11 后可扩到 2024。

用法
----
  .\venv\Scripts\python.exe src\preprocess\ndvi_drought_render_json.py --limit-months 2
  .\venv\Scripts\python.exe src\preprocess\ndvi_drought_render_json.py
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
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
NDVI_DIR = os.path.join(ROOT, "data", "raw", "01_optical_vnir", "modis_ndvi")
SPEI = os.path.join(ROOT, "data", "raw", "spei", "spei03.nc")
OUT = os.path.join(ROOT, "datasets", "01_optical_vnir", "drought_ndvi")
IMG = os.path.join(OUT, "images")

WIN = 128              # 0.05° × 128 = 6.4°
STRIDE = 128
MIN_VALID = 0.5
POS_PER_CLASS = 6      # 每月每类最多几个窗
SEED = 42
OUT_SIZE = 384
SCALE = 0.0001
FILL = -3000

PROMPT = ("This is a MODIS NDVI anomaly map over a region, in percent of the same-month "
          "climatology. Positive = greener, negative = browning/water stress. "
          "Output a JSON object with fields: disaster_type, detected (true/false), "
          "source, physical_quantity, severity, mean_ndvi_anomaly_pct, criterion. "
          "severity must be one of these exact words: normal / slight / moderate / severe.")


def load_spei():
    ds = Dataset(SPEI)
    spei = np.ma.filled(np.array(ds.variables["spei"][:], dtype="float32"), 1e30)
    time = np.array(ds.variables["time"][:], dtype="float64")
    lat = np.array(ds.variables["lat"][:], dtype="float64")
    lon = np.array(ds.variables["lon"][:], dtype="float64")
    ds.close()
    spei[spei > 1e20] = np.nan
    return spei, time, lat, lon


def spei_index(year, month):
    return (year - 1901) * 12 + (month - 1)


def severity_of(v):
    if v > -0.5:
        return "normal"
    if v > -1.0:
        return "slight"
    if v > -1.5:
        return "moderate"
    return "severe"


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="RdBu_r", vmin=-15, vmax=15, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("NDVI anomaly (%)", fontsize=7)
    fig.text(0.04, 0.075, "MODIS MOD13C2, monthly 0.05 deg", fontsize=6)
    fig.text(0.04, 0.035, "positive = greener than same-month climatology", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-months", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = {}
    for f in glob.glob(os.path.join(NDVI_DIR, "*.hdf")):
        m = re.search(r"A(\d{4})(\d{3})", os.path.basename(f))
        if not m:
            continue
        y, doy = int(m.group(1)), int(m.group(2))
        import datetime
        month = (datetime.date(y, 1, 1) + datetime.timedelta(days=doy - 1)).month
        files[(y, month)] = f

    spei, stime, slat, slon = load_spei()
    max_spei_time = stime[-1]
    import datetime
    max_date = datetime.date(1900, 1, 1) + datetime.timedelta(days=float(max_spei_time))
    print("NDVI 月份 %d；SPEI 覆盖到 %s" % (len(files), max_date))

    months = sorted(files)
    if args.limit_months:
        months = months[:args.limit_months]

    ndvi_cache = {}

    def load_ndvi(y, mo):
        if (y, mo) in ndvi_cache:
            return ndvi_cache[(y, mo)]
        sd = SD(files[(y, mo)], SDC.READ)
        ndvi = np.array(sd.select("CMG 0.05 Deg Monthly NDVI")[:], dtype="float32") / 10000.0
        qa = np.array(sd.select("CMG 0.05 Deg Monthly pixel reliability")[:])
        sd.end()
        ndvi[(qa != 0)] = np.nan      # 只用最好质量（0=good）
        ndvi[(ndvi < -0.2) | (ndvi > 1.0)] = np.nan
        ndvi_cache[(y, mo)] = ndvi
        return ndvi

    records = []
    random.seed(SEED)
    for (y, mo) in months:
        if spei_index(y, mo) > spei_index(max_date.year, max_date.month):
            continue
        cur = load_ndvi(y, mo)
        h, w = cur.shape
        clim = []
        for yy in range(2020, 2025):
            if yy == y or (yy, mo) not in files:
                continue
            clim.append(load_ndvi(yy, mo))
        if not clim:
            continue
        base = np.nanmean(np.stack(clim), axis=0)
        anom = np.where(base > 0.05, (cur - base) / np.maximum(base, 1e-6) * 100.0, np.nan)
        si = spei_index(y, mo)
        sp = spei[si]
        cands = {True: [], False: []}
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                a = anom[r0:r0 + WIN, c0:c0 + WIN]
                fin = np.isfinite(a)
                if fin.mean() < MIN_VALID:
                    continue
                la = 89.975 - (r0 + WIN / 2) * 0.05
                lo = -179.975 + (c0 + WIN / 2) * 0.05
                r = int(np.clip(round((la + 89.75) / 0.5), 0, 359))   # SPEI 网格南→北
                c = int(np.clip(round((lo + 179.75) / 0.5), 0, 719))
                s = float(sp[max(0, r - 1):r + 2, max(0, c - 1):c + 2].mean())
                if not np.isfinite(s):
                    continue
                det = s <= -1.0
                cands[det].append((r0, c0, la, lo, float(np.nanmean(a)), s))
        for det in (True, False):
            lst = cands[det]
            random.shuffle(lst)
            for (r0, c0, la, lo, mean_a, s) in lst[:POS_PER_CLASS]:
                sev = severity_of(s)
                name = "drought_%04d%02d_r%d_c%d.png" % (y, mo, r0, c0)
                p = os.path.join(IMG, name)
                render(anom[r0:r0 + WIN, c0:c0 + WIN], p, "%04d-%02d" % (y, mo), la, lo)
                ans = {
                    "disaster_type": "drought",
                    "detected": bool(det),
                    "source": "OPT",
                    "physical_quantity": "ndvi_anomaly",
                    "severity": sev,
                    "mean_ndvi_anomaly_pct": round(mean_a, 2),
                    "criterion": ("SPEI-3 <= -1.0 = moderate drought (McKee et al. 1993); "
                                  "NDVI anomaly in percent of same-month 2020-2024 climatology"),
                }
                records.append({
                    "image": p, "label": sev, "stamp": "%04d-%02d" % (y, mo),
                    "mean_ndvi_anomaly_pct": round(mean_a, 2),
                    "conversations": [
                        {"from": "human", "value": "<image>" + PROMPT},
                        {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                    ],
                })
        print("  %04d-%02d  +%d/%d  累计 %d" % (y, mo, min(len(cands[True]), POS_PER_CLASS),
                                               min(len(cands[False]), POS_PER_CLASS), len(records)))

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

    json.dump(train, open(os.path.join(OUT, "train_drought_ndvi_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_drought_ndvi_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
