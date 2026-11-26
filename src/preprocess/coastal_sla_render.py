# -*- coding: utf-8 -*-
r"""
海岸高水位 × 测高（CMEMS L4 SLA 百分位）切窗 + 结构化 JSON（新元素：coastal_sla）

物理链路
--------
  多卫星测高（Sentinel-3/6、Jason-3、Saral…）→ 海面高度异常 SLA（m）
  SLA 百分位高 = 沿岸水位异常偏高 → 海岸洪水/风暴潮风险
任务
----
  detected：窗内平均 SLA 百分位 ≥ 90
  数值字段：mean_sla_percentile（0–100，图上可读）
  severity：none <90 / light 90–95 / moderate 95–98 / heavy ≥98

用法
----
  .\venv\Scripts\python.exe src\preprocess\coastal_sla_render.py --step 3
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

import numpy as np
from netCDF4 import Dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "SEALEVEL_GLO_PHY_L4_MY_008_047")
OUT = os.path.join(ROOT, "datasets", "08_altimetry", "coastal_sla")
IMG = os.path.join(OUT, "images")

WIN = 16
STRIDE = 8
MIN_VALID = 0.3
POS_PER_DAY = 6
NEG_PER_DAY = 6
SEED = 42
OUT_SIZE = 384
LATB = (15.0, 42.0)
LONB = (105.0, 135.0)

PROMPT = ("This is a satellite altimetry sea-level-anomaly percentile map (0-100) around the China "
          "seas: high values mean water level is unusually high and may cause coastal flooding. "
          "Output a JSON object with fields: disaster_type, detected (true/false), source, "
          "physical_quantity, severity, mean_sla_percentile, criterion. severity must be one of "
          "these exact words: none / light / moderate / heavy.")


def severity_of(p):
    if p < 90:
        return "none"
    if p < 95:
        return "light"
    if p < 98:
        return "moderate"
    return "heavy"


def load_sla(fp):
    ds = Dataset(fp)
    sla = np.array(ds.variables["sla"][0], dtype="float32")
    lat = np.array(ds.variables["latitude"][:], dtype="float64")
    lon = np.array(ds.variables["longitude"][:], dtype="float64")
    t = ds.variables["time"]
    try:
        import cftime
        stamp = str(cftime.num2date(float(t[0]), t.units, t.calendar))[:10].replace("-", "")
    except Exception:
        stamp = str(float(t[0]))[:10]
    ds.close()
    sla[(sla < -10) | (sla > 10)] = np.nan
    return sla, lat, lon, stamp


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("SLA percentile", fontsize=7)
    fig.text(0.04, 0.075, "CMEMS L4 altimetry, 0.25 deg", fontsize=6)
    fig.text(0.04, 0.035, "white = no data", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(RAW, "*.nc")))[::a.step]
    if a.limit:
        files = files[:a.limit]
    print("SLA 日文件 %d 个" % len(files))

    stack, lat0, lon0, stamps = [], None, None, []
    for fp in files:
        sla, lat, lon, stamp = load_sla(fp)
        stack.append(sla)
        if lat0 is None:
            lat0, lon0 = lat, lon
        stamps.append(stamp)
    stack = np.stack(stack)
    print("堆叠", stack.shape)

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
                la = float(np.nanmean(lat0[r0:r0 + WIN]))
                lo = float(np.nanmean(lon0[c0:c0 + WIN]))
                if not (LATB[0] < la < LATB[1] and LONB[0] < lo < LONB[1]):
                    continue
                mp = float(np.nanmean(wp))
                item = (r0, c0, la, lo, mp)
                if mp >= 90:
                    pos.append(item)
                elif mp < 40:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for det, (r0, c0, la, lo, mp) in [(True, x) for x in pos[:POS_PER_DAY]] + \
                [(False, x) for x in neg[:NEG_PER_DAY]]:
            sev = severity_of(mp)
            nm = "sla_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            render(p[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "coastal_high_water",
                "detected": bool(det),
                "source": "ALT",
                "physical_quantity": "sea_level_anomaly_percentile",
                "severity": sev,
                "mean_sla_percentile": round(mp, 1),
                "criterion": ("SLA percentile >= 90 = unusually high coastal water level (study-defined); "
                              "percentile over the daily SLA subset"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "mean_sla_percentile": round(mp, 1),
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
        n = min(max(2, int(round(len(lst) * 0.2))), 40, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_coastal_sla_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_coastal_sla_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
