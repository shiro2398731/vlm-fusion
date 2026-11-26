# -*- coding: utf-8 -*-
r"""
干旱 × CHIRPS 降水百分位（月，150 个月 2010–2022）二分类（新元素：drought_precip_pct）

逐像元百分位：低百分位 = 该月降水显著偏少（气象干旱）。
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
from netCDF4 import Dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "06_radar_precipitation", "chirps_monthly")
OUT = os.path.join(ROOT, "datasets", "06_radar_precipitation", "drought_precip_pct")
IMG = os.path.join(OUT, "images")

WIN = 16
STRIDE = 8
MIN_VALID = 0.5
POS_PER_MONTH = 6
NEG_PER_MONTH = 6
SEED = 42
OUT_SIZE = 384
LATB = (18.0, 54.0)
LONB = (73.0, 135.0)

PROMPT = ("This is a CHIRPS monthly precipitation percentile map (0-100) over China: low values "
          "mean rainfall is unusually low. Output a JSON object with fields: disaster_type, "
          "detected (true/false), source, physical_quantity, severity, criterion. severity must "
          "be one of these exact words: none / drought.")


def load_china(fp):
    ds = Dataset(fp)
    p = np.ma.filled(np.array(ds.variables["precip"][:], dtype="float32"), np.nan)
    lat = np.array(ds.variables["latitude"][:], dtype="float64")
    lon = np.array(ds.variables["longitude"][:], dtype="float64")
    ds.close()
    if p.ndim == 3:
        p = p[0]
    p[(p < 0) | (p > 3000)] = np.nan
    rmask = (lat >= LATB[0]) & (lat <= LATB[1])
    cmask = (lon >= LONB[0]) & (lon <= LONB[1])
    return p[rmask][:, cmask], lat[rmask], lon[cmask]


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="YlOrBr", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("precip percentile", fontsize=7)
    fig.text(0.04, 0.075, "CHIRPS monthly, 0.05 deg", fontsize=6)
    fig.text(0.04, 0.035, "light = dry", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = {}
    for f in glob.glob(os.path.join(RAW, "*.nc")):
        m = re.search(r"(\d{4})(\d{2})", os.path.basename(f))
        if m:
            files[(int(m.group(1)), int(m.group(2)))] = f
    keys = sorted(files)
    if a.limit:
        keys = keys[:a.limit]
    print("月文件 %d" % len(keys))

    stack, lat, lon, stamps = [], None, None, []
    for (y, mo) in keys:
        p, la, lo = load_china(files[(y, mo)])
        stack.append(p)
        if lat is None:
            lat, lon = la, lo
        stamps.append("%d%02d" % (y, mo))
    stack = np.stack(stack)
    print("堆叠", stack.shape)

    pct = np.full_like(stack, np.nan)
    D = stack.shape[0]
    for i in range(D):
        v = stack[i]
        lower = np.nansum(stack < v, axis=0)
        valid = np.sum(np.isfinite(stack), axis=0)
        pct[i] = np.where(valid >= 10, 100.0 * lower / np.maximum(valid, 1), np.nan)

    h, w = pct.shape[1], pct.shape[2]
    records = []
    random.seed(SEED)
    for i, stamp in enumerate(stamps, 1):
        p = pct[i - 1]
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                wp = p[r0:r0 + WIN, c0:c0 + WIN]
                if np.isfinite(wp).mean() < MIN_VALID:
                    continue
                mp = float(np.nanmean(wp))
                item = (r0, c0, float(lat[r0 + WIN // 2]), float(lon[c0 + WIN // 2]), mp)
                if mp <= 10:
                    pos.append(item)
                elif mp >= 40:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for det, (r0, c0, la, lo, mp) in [(True, x) for x in pos[:POS_PER_MONTH]] + \
                [(False, x) for x in neg[:NEG_PER_MONTH]]:
            sev = "drought" if det else "none"
            nm = "dpp_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            render(p[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "drought",
                "detected": bool(det),
                "source": "PRECIP",
                "physical_quantity": "precip_percentile",
                "severity": sev,
                "criterion": "monthly precipitation percentile <= 10 = meteorological drought",
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if i % 50 == 0:
            print("  [%d/%d] %s 累计 %d" % (i, len(stamps), stamp, len(records)))

    seen = set()
    uniq = []
    for r in records:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        uniq.append(r)
    random.seed(SEED)
    by = {}
    for r in uniq:
        by.setdefault(r["label"], []).append(r)
    val, train = [], []
    for k, lst in sorted(by.items()):
        lst = lst[:]
        random.shuffle(lst)
        n = min(max(2, int(round(len(lst) * 0.2))), 60, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])
    json.dump(train, open(os.path.join(OUT, "train_drought_precip_pct_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_drought_precip_pct_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
