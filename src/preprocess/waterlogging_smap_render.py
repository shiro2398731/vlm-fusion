# -*- coding: utf-8 -*-
r"""
干旱 × 微波（SMAP L3 土壤湿度）切窗 + 结构化 JSON（新元素：waterlogging_smap）

物理链路
--------
  SMAP L 波段辐射计 → 亮度温度 → 土壤湿度（m³/m³，表层 5cm）
  低土壤湿度 = 气象/农业干旱的直接观测量
真值
----
  SPEI-3 ≤ -1.0（McKee et al. 1993）
任务
----
  detected：窗口 SPEI-3 是否 ≤ -1.0（真值）
  数值字段：mean_soil_moisture（m³/m³，只能从图上读）
  severity：none <0.30 / light 0.30–0.38 / moderate 0.38–0.45 / heavy >0.45

用法
----
  .\venv\Scripts\python.exe src\preprocess\waterlogging_smap_render.py --limit-days 3
  .\venv\Scripts\python.exe src\preprocess\waterlogging_smap_render.py --limit-days 60
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
import h5py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "smap2")
SPEI = os.path.join(ROOT, "data", "raw", "spei", "spei03.nc")
OUT = os.path.join(ROOT, "datasets", "04_passive_microwave", "waterlogging_smap")
IMG = os.path.join(OUT, "images")

WIN = 16
STRIDE = 8
MIN_VALID = 0.3
POS_PER_DAY = 6
NEG_PER_DAY = 6
SEED = 42
OUT_SIZE = 384
LATB = (15.0, 48.0)
LONB = (75.0, 135.0)

PROMPT = ("This is a SMAP satellite surface soil-moisture map (m3/m3) over China. "
          "High soil moisture indicates waterlogging. Output a JSON object with fields: "
          "disaster_type, detected (true/false), source, physical_quantity, severity, "
          "mean_soil_moisture, criterion. severity must be one of these exact words: "
          "none / light / moderate / heavy.")


def severity_of(sm):
    if sm < 0.30:
        return "none"
    if sm < 0.38:
        return "light"
    if sm < 0.45:
        return "moderate"
    return "heavy"

def spei_index(y, m):
    return (y - 1901) * 12 + (m - 1)


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="YlGnBu", vmin=0, vmax=0.5, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("soil moisture (m3/m3)", fontsize=6.5)
    fig.text(0.04, 0.075, "SMAP L3, 36 km grid", fontsize=6)
    fig.text(0.04, 0.035, "white = no retrieval", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def load_smap(fp):
    with h5py.File(fp, "r") as h:
        g = h["Soil_Moisture_Retrieval_Data_AM"]
        sm = g["soil_moisture"][:].astype("float32")
        lat = g["latitude"][:].astype("float64")
        lon = g["longitude"][:].astype("float64")
        try:
            ft = g["freeze_thaw_fraction"][:]
        except KeyError:
            ft = None
    sm[(sm < 0) | (sm > 1)] = np.nan
    lat[(lat < -90) | (lat > 90)] = np.nan
    lon[(lon < -180) | (lon > 180)] = np.nan
    if ft is not None:
        ft = ft.astype("float32")
        ft[(ft < 0) | (ft > 1)] = np.nan
        sm[np.isfinite(ft) & (ft > 0.5)] = np.nan   # 冻土不判干旱
    return sm, lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    ds = Dataset(SPEI)
    sp = np.ma.filled(np.array(ds.variables["spei"][:], dtype="float32"), 1e30)
    ds.close()
    sp[sp > 1e20] = np.nan

    files = sorted(glob.glob(os.path.join(RAW, "SMAP_L3_SM_P_*.h5")))
    print("SMAP 日文件 %d 个" % len(files))
    if a.limit_days:
        files = files[:a.limit_days]

    records = []
    random.seed(SEED)
    for fi, fp in enumerate(files, 1):
        m = re.search(r"SMAP_L3_SM_P_(\d{8})", os.path.basename(fp))
        if not m:
            continue
        d = datetime.date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:]))
        sm, lat, lon = load_smap(fp)
        h, w = sm.shape
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                wsm = sm[r0:r0 + WIN, c0:c0 + WIN]
                if np.isfinite(wsm).mean() < MIN_VALID:
                    continue
                la = float(np.nanmean(lat[r0:r0 + WIN, c0:c0 + WIN]))
                lo = float(np.nanmean(lon[r0:r0 + WIN, c0:c0 + WIN]))
                if not (LATB[0] < la < LATB[1] and LONB[0] < lo < LONB[1]):
                    continue
                msm = float(np.nanmean(wsm))
                item = (r0, c0, la, lo, msm, 0.0)
                if msm >= 0.38:
                    pos.append(item)
                elif msm < 0.22:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        stamp = m.group(1)
        for det, (r0, c0, la, lo, msm, s) in [(True, x) for x in pos[:POS_PER_DAY]] + \
                [(False, x) for x in neg[:NEG_PER_DAY]]:
            sev = severity_of(msm)
            nm = "wlm_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            render(sm[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "waterlogging",
                "detected": bool(det),
                "source": "MW",
                "physical_quantity": "soil_moisture",
                "severity": sev,
                "mean_soil_moisture": round(msm, 3),
                "criterion": ("SPEI-3 <= -1.0 = moderate drought (McKee 1993); "
                              "SMAP L3 surface soil moisture; severity bands are soil-moisture based"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "mean_soil_moisture": round(msm, 3),
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

    json.dump(train, open(os.path.join(OUT, "train_waterlogging_smap_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_waterlogging_smap_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
