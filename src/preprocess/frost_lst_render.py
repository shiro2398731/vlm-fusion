# -*- coding: utf-8 -*-
r"""
热浪 × TIR（MODIS MOD11C3 月 LST 百分位）二分类（新元素：frost_lst）

逐像元百分位（36 个月）：高百分位 = 地表温度异常偏高（热浪）。
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np
from pyhdf.SD import SD, SDC

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "03_thermal_infrared", "mod11c3")
OUT = os.path.join(ROOT, "datasets", "03_thermal_infrared", "frost_lst")
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
RES = 0.05

PROMPT = ("This is a MODIS land-surface-temperature percentile map (0-100): low values mean the "
          "surface is unusually cold. Output a JSON object with fields: disaster_type, detected "
          "(true/false), source, physical_quantity, severity, criterion. severity must be one "
          "of these exact words: none / frost.")


def load_china(fp):
    sd = SD(fp, SDC.READ)
    lst = np.array(sd.select("LST_Day_CMG")[:], dtype="float32") * 0.02
    sd.end()
    lst[(lst < 200) | (lst > 360)] = np.nan
    r0 = int((90 - LATB[1]) / RES)
    r1 = int((90 - LATB[0]) / RES)
    c0 = int((LONB[0] + 180) / RES)
    c1 = int((LONB[1] + 180) / RES)
    return lst[r0:r1, c0:c1]


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="Blues", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("LST percentile", fontsize=7)
    fig.text(0.04, 0.075, "MODIS MOD11C3 monthly, 0.05 deg", fontsize=6)
    fig.text(0.04, 0.035, "dark blue = unusually cold", fontsize=6, color="#666")
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
    for f in glob.glob(os.path.join(RAW, "*.hdf")):
        m = re.search(r"\.A(\d{7})\.", os.path.basename(f))
        if m:
            files[m.group(1)] = f
    keys = sorted(files)
    if a.limit:
        keys = keys[:a.limit]
    print("月文件 %d" % len(keys))

    stack, stamps = [], []
    for k in keys:
        stack.append(load_china(files[k]))
        stamps.append(k)
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
    lats = LATB[1] - np.arange(h) * RES
    lons = LONB[0] + np.arange(w) * RES

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
                item = (r0, c0, float(lats[r0 + WIN // 2]), float(lons[c0 + WIN // 2]), mp)
                if mp <= 10:
                    pos.append(item)
                elif mp >= 50:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for det, (r0, c0, la, lo, mp) in [(True, x) for x in pos[:POS_PER_MONTH]] + \
                [(False, x) for x in neg[:NEG_PER_MONTH]]:
            sev = "frost" if det else "none"
            nm = "hw_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            render(p[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "frost",
                "detected": bool(det),
                "source": "TIR",
                "physical_quantity": "lst_percentile",
                "severity": sev,
                "criterion": "LST percentile >= 90 = unusually hot surface (frost proxy)",
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if i % 10 == 0:
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
    json.dump(train, open(os.path.join(OUT, "train_frost_lst_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_frost_lst_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
