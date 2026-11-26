# -*- coding: utf-8 -*-
r"""
干旱 × 降水（CHIRPS 月降水）切窗 + 结构化 JSON（新元素：drought_precip）

物理链路
--------
  CHIRPS 月降水（0.05°，这里用 0.1°）→ 月降水距平（相对 2010–2019 同月气候态）
  → 降水亏缺 = 气象干旱
真值
----
  SPEI-3 ≤ -1.0（McKee et al. 1993；含蒸散）
任务
----
  detected：窗口是否处于 SPEI-3 ≤ -1.0 的干旱区
  数值字段：mean_precip_anomaly（mm/月，只能从图上读）
  severity：normal > -0.5 / slight -0.5~-1 / moderate -1~-1.5 / severe ≤ -1.5

用法
----
  .\venv\Scripts\python.exe src\preprocess\drought_precip_monthly_render.py --limit-months 2
  .\venv\Scripts\python.exe src\preprocess\drought_precip_monthly_render.py
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
SPEI = os.path.join(ROOT, "data", "raw", "spei", "spei03.nc")
OUT = os.path.join(ROOT, "datasets", "06_radar_precipitation", "drought_precip")
IMG = os.path.join(OUT, "images")

WIN = 64
STRIDE = 32
MIN_VALID = 0.5
POS_PER_MONTH = 8
NEG_PER_MONTH = 6
SEED = 42
OUT_SIZE = 384
CLIM0, CLIM1 = 2010, 2019

PROMPT = ("This is a 3-month accumulated precipitation anomaly map over China (mm per 3 months) versus the "
          "2010-2019 climatology: negative = rainfall deficit. Output a JSON object with "
          "fields: disaster_type, detected (true/false), source, physical_quantity, "
          "severity, mean_precip_anomaly, criterion. severity must be one of these exact "
          "words: normal / slight / moderate / severe.")


def severity_of(spei):
    if spei > -0.5:
        return "normal"
    if spei > -1.0:
        return "slight"
    if spei > -1.5:
        return "moderate"
    return "severe"


def spei_index(y, m):
    return (y - 1901) * 12 + (m - 1)


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="RdBu", vmin=-120, vmax=120, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("3-month precip anomaly (mm)", fontsize=6.5)
    fig.text(0.04, 0.075, "CHIRPS monthly, 0.1 deg grid", fontsize=6)
    fig.text(0.04, 0.035, "red = deficit, blue = surplus", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-months", type=int, default=0)
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
    print("降水月份 %d，格点 %s" % (len(data), data[(2020, 1)].shape))

    ds = Dataset(SPEI)
    sp = np.ma.filled(np.array(ds.variables["spei"][:], dtype="float32"), 1e30)
    ds.close()
    sp[sp > 1e20] = np.nan

    months = sorted([k for k in data if 2020 <= k[0] <= 2022])
    if a.limit_months:
        months = months[:a.limit_months]

    records = []
    random.seed(SEED)
    for (y, mo) in months:
        clim = [data[(yy, mo)] for yy in range(CLIM0, CLIM1 + 1) if (yy, mo) in data]
        if len(clim) < 5:
            continue
        base = np.nanmean(np.stack(clim), axis=0)
        anom = data[(y, mo)] - base
        si = spei_index(y, mo)
        if si >= sp.shape[0]:
            continue
        spm = sp[si]
        h, w = anom.shape
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                wa = anom[r0:r0 + WIN, c0:c0 + WIN]
                if np.isfinite(wa).mean() < MIN_VALID:
                    continue
                la = float(np.nanmean(lat[r0:r0 + WIN]))
                lo = float(np.nanmean(lon[c0:c0 + WIN]))
                r = int(np.clip(round((la + 89.75) / 0.5), 0, 359))
                c = int(np.clip(round((lo + 179.75) / 0.5), 0, 719))
                s = float(np.nanmean(spm[max(0, r - 1):r + 2, max(0, c - 1):c + 2]))
                if not np.isfinite(s):
                    continue
                item = (r0, c0, la, lo, float(np.nanmean(wa)), s)
                if s <= -1.0:
                    pos.append(item)
                elif s > -0.5:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        stamp = "%04d%02d" % (y, mo)
        for det, (r0, c0, la, lo, mean_a, s) in [(True, x) for x in pos[:POS_PER_MONTH]] + \
                [(False, x) for x in neg[:NEG_PER_MONTH]]:
            sev = severity_of(s)
            nm = "dprecipm_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            render(anom[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "drought",
                "detected": bool(det),
                "source": "PRECIP",
                "physical_quantity": "precip_anomaly_3mo",
                "severity": sev,
                "mean_precip_anomaly_3mo": round(mean_a, 1),
                "criterion": ("SPEI-3 <= -1.0 = moderate drought (McKee 1993); "
                              "3-month CHIRPS anomaly vs 2010-2019 climatology"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "mean_precip_anomaly_3mo": round(mean_a, 1),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        print("  %s 累计 %d" % (stamp, len(records)))

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

    json.dump(train, open(os.path.join(OUT, "train_drought_precip_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_drought_precip_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
