# -*- coding: utf-8 -*-
r"""
渍涝 × 微波（SMAP 土壤湿度百分位）切窗 + 结构化 JSON（新元素：waterlogging_pct）

为什么用百分位
--------------
  SMAP 36km 绝对土壤湿度在窗口内太均匀，模型读不出差异；
  改为「每个像元在本数据集 46 个日期中的百分位」（0–100），
  高百分位 = 异常偏湿 → 渍涝风险。图像对比度显著提高，且数值仍可从图读。

任务
----
  detected：窗内平均百分位 ≥ 80（异常偏湿）
  数值字段：mean_sm_percentile（0–100，图上可读）
  severity：none <80 / light 80–90 / moderate 90–97 / heavy ≥97

用法
----
  .\venv\Scripts\python.exe src\preprocess\waterlogging_pct_render.py
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
RAW = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "smap2")
OUT = os.path.join(ROOT, "datasets", "04_passive_microwave", "waterlogging_pct")
IMG = os.path.join(OUT, "images")

WIN = 8
STRIDE = 8
MIN_VALID = 0.3
POS_PER_DAY = 8
NEG_PER_DAY = 8
SEED = 42
OUT_SIZE = 384
LATB = (15.0, 48.0)
LONB = (75.0, 135.0)

PROMPT = ("This is a SMAP soil-moisture percentile map (0-100) over China: high values mean the "
          "ground is unusually wet for this dataset and may be waterlogged. Output a JSON "
          "object with fields: disaster_type, detected (true/false), source, "
          "physical_quantity, severity, mean_sm_percentile, criterion. severity must be one "
          "of these exact words: none / light / moderate / heavy.")


def severity_of(p):
    if p < 80:
        return "none"
    if p < 90:
        return "light"
    if p < 97:
        return "moderate"
    return "heavy"


def load_sm(fp):
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
        sm[np.isfinite(ft) & (ft > 0.5)] = np.nan
    return sm, lat, lon


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("soil moisture percentile", fontsize=6.5)
    fig.text(0.04, 0.075, "SMAP L3, percentile over 46 dates", fontsize=6)
    fig.text(0.04, 0.035, "white = no retrieval", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(RAW, "SMAP_L3_SM_P_*.h5")))
    print("SMAP 日文件 %d 个" % len(files))
    if a.limit_days:
        files = files[:a.limit_days]

    sm_list, lat, lon, stamps = [], None, None, []
    for fp in files:
        m = re.search(r"SMAP_L3_SM_P_(\d{8})", os.path.basename(fp))
        sm, la, lo = load_sm(fp)
        sm_list.append(sm)
        if lat is None:
            lat, lon = la, lo
        stamps.append(m.group(1))
    stack = np.stack(sm_list)          # (D, H, W)
    print("堆叠", stack.shape)

    # 逐像元百分位
    pct = np.full_like(stack, np.nan)
    D = stack.shape[0]
    for i in range(D):
        v = stack[i]
        lower = np.nansum(stack < v, axis=0)
        valid = np.sum(np.isfinite(stack), axis=0)
        pct[i] = np.where(valid >= 5, 100.0 * lower / np.maximum(valid, 1), np.nan)

    records = []
    random.seed(SEED)
    for i, stamp in enumerate(stamps, 1):
        p = pct[i - 1]
        h, w = p.shape
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                wp = p[r0:r0 + WIN, c0:c0 + WIN]
                if np.isfinite(wp).mean() < MIN_VALID:
                    continue
                la = float(np.nanmean(lat[r0:r0 + WIN, c0:c0 + WIN]))
                lo = float(np.nanmean(lon[r0:r0 + WIN, c0:c0 + WIN]))
                if not (LATB[0] < la < LATB[1] and LONB[0] < lo < LONB[1]):
                    continue
                mp = float(np.nanmax(wp))
                item = (r0, c0, la, lo, mp)
                if mp >= 85:
                    pos.append(item)
                elif mp < 50:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for det, (r0, c0, la, lo, mp) in [(True, x) for x in pos[:POS_PER_DAY]] + \
                [(False, x) for x in neg[:NEG_PER_DAY]]:
            sev = severity_of(mp)
            nm = "wlp_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            render(p[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "waterlogging",
                "detected": bool(det),
                "source": "MW",
                "physical_quantity": "sm_percentile",
                "severity": sev,
                "mean_sm_percentile": round(mp, 1),
                "criterion": ("soil moisture percentile >= 80 = unusually wet / waterlogging risk "
                              "(study-defined); percentile over the 46-date SMAP subset"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "mean_sm_percentile": round(mp, 1),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if i % 10 == 0 or i == len(stamps):
            print("  [%d/%d] %s 累计 %d" % (i, len(stamps), stamp, len(records)))

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

    json.dump(train, open(os.path.join(OUT, "train_waterlogging_pct_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_waterlogging_pct_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
