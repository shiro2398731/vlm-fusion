# -*- coding: utf-8 -*-
r"""
珊瑚白化（DHW）切窗 + 结构化 JSON（新元素：coral_dhw）

物理链路
--------
  卫星 SST（CRW 5km 日产品）→ 海温异常累积 → 白化热周指数 DHW（°C-weeks）
真值/判据
---------
  NOAA Coral Reef Watch：DHW >= 4 °C-weeks = 白化预警 1 级（Bleaching Alert Level 1）
  这是国际业务标准，属于「①类」判据
任务
----
  detected：窗内最大 DHW 是否 >= 4（读数 + 阈值）
  数值字段：max_dhw（°C-weeks）
  severity：none<1 / watch 1-4 / alert1 4-8 / alert2>=8

用法
----
  .\venv\Scripts\python.exe src\preprocess\coral_dhw_render_json.py --top-days 5
  .\venv\Scripts\python.exe src\preprocess\coral_dhw_render_json.py --top-days 120
"""
import argparse
import collections
import datetime
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
RAW = os.path.join(ROOT, "data", "raw", "03_thermal_infrared", "coral_dhw")
OUT = os.path.join(ROOT, "datasets", "03_thermal_infrared", "coral_dhw")
IMG = os.path.join(OUT, "images")

WIN = 32
STRIDE = 16
MIN_VALID = 0.5
POS_PER_DAY = 4
NEG_PER_DAY = 4
ALERT = 4.0
SEED = 42
OUT_SIZE = 384

PROMPT = ("This is a NOAA Coral Reef Watch Degree Heating Weeks (DHW) map over a coral reef "
          "region, in degC-weeks. DHW >= 4 indicates a bleaching alert level 1. Output a "
          "JSON object with fields: disaster_type, detected (true/false), source, "
          "physical_quantity, severity, max_dhw, criterion. severity must be one of these "
          "exact words: none / watch / alert1 / alert2.")


def severity_of(v):
    if v < 1:
        return "none"
    if v < 4:
        return "watch"
    if v < 8:
        return "alert1"
    return "alert2"


def load(fp):
    ds = Dataset(fp)
    a = np.ma.filled(np.array(ds.variables["CRW_DHW"][:], dtype="float32"), -999.0)
    lat = np.array(ds.variables["latitude"][:], dtype="float64")
    lon = np.array(ds.variables["longitude"][:], dtype="float64")
    times = ds.variables["time"][:]
    ds.close()
    a[(a < -100) | (a > 50)] = np.nan
    return a, lat, lon, times


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="YlOrRd", vmin=0, vmax=8, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("DHW (degC-weeks)", fontsize=7)
    fig.text(0.04, 0.075, "NOAA Coral Reef Watch, daily 5 km", fontsize=6)
    fig.text(0.04, 0.035, "white = land / no data", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-days", type=int, default=120)
    a = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(RAW, "chla_*.nc")))
    print("DHW 月块 %d 个" % len(files))
    if not files:
        return

    # 第一遍：每天最佳窗的 max DHW
    day_best = {}
    meta = {}
    for f in files:
        arr_all, lat, lon, times = load(f)
        for i in range(arr_all.shape[0]):
            stamp = str(datetime.datetime.utcfromtimestamp(float(times[i])))[:10].replace("-", "")
            arr = arr_all[i]
            h, w = arr.shape
            best = 0.0
            for r0 in range(0, h - WIN + 1, STRIDE):
                for c0 in range(0, w - WIN + 1, STRIDE):
                    ww = arr[r0:r0 + WIN, c0:c0 + WIN]
                    if np.isfinite(ww).mean() < MIN_VALID:
                        continue
                    v = float(np.nanmax(ww))
                    if v > best:
                        best = v
            day_best[stamp] = best
            meta[stamp] = f
    days = sorted([d for d, v in day_best.items() if v >= ALERT], key=lambda d: -day_best[d])
    if a.top_days:
        days = days[:a.top_days]
    print("白化预警日 %d 个（取 %d）" % (len([d for d, v in day_best.items() if v >= ALERT]), len(days)))
    if not days:
        print("没有白化预警日")
        return

    by_file = collections.defaultdict(list)
    for d in days:
        by_file[meta[d]].append(d)
    records = []
    random.seed(SEED)
    done = 0
    for f, ds_days in by_file.items():
        arr_all, lat, lon, times = load(f)
        for i in range(arr_all.shape[0]):
            stamp = str(datetime.datetime.utcfromtimestamp(float(times[i])))[:10].replace("-", "")
            if stamp not in ds_days:
                continue
            arr = arr_all[i]
            h, w = arr.shape
            pos, neg = [], []
            for r0 in range(0, h - WIN + 1, STRIDE):
                for c0 in range(0, w - WIN + 1, STRIDE):
                    ww = arr[r0:r0 + WIN, c0:c0 + WIN]
                    if np.isfinite(ww).mean() < MIN_VALID:
                        continue
                    mx = float(np.nanmax(ww))
                    la = float(np.nanmean(lat[r0:r0 + WIN]))
                    lo = float(np.nanmean(lon[c0:c0 + WIN]))
                    if mx >= ALERT:
                        pos.append((r0, c0, la, lo, mx))
                    elif mx < 1.0:
                        neg.append((r0, c0, la, lo, mx))
            random.shuffle(pos)
            random.shuffle(neg)
            for (r0, c0, la, lo, mx) in pos[:POS_PER_DAY] + neg[:NEG_PER_DAY]:
                sev = severity_of(mx)
                det = mx >= ALERT
                nm = "coral_%s_r%d_c%d.png" % (stamp, r0, c0)
                p = os.path.join(IMG, nm)
                render(arr[r0:r0 + WIN, c0:c0 + WIN], p, stamp, la, lo)
                ans = {
                    "disaster_type": "coral_bleaching",
                    "detected": bool(det),
                    "source": "TIR",
                    "physical_quantity": "degree_heating_weeks",
                    "severity": sev,
                    "max_dhw": round(mx, 2),
                    "criterion": ("DHW >= 4 degC-weeks = NOAA CRW Bleaching Alert Level 1; "
                                  "severity bands: none/watch/alert1/alert2"),
                }
                records.append({
                    "image": p, "label": sev, "stamp": stamp, "max_dhw": round(mx, 2),
                    "conversations": [
                        {"from": "human", "value": "<image>" + PROMPT},
                        {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                    ],
                })
            done += 1
            if done % 10 == 0:
                print("  %d/%d 天 累计 %d" % (done, len(days), len(records)))

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

    json.dump(train, open(os.path.join(OUT, "train_coral_dhw_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_coral_dhw_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
