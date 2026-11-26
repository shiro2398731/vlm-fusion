# -*- coding: utf-8 -*-
r"""
地下水 × GRACE-DA（GRACEDADM_CLSM025GL_7D，7 天 0.25°）二分类（新元素：waterlogging_grace）

物理链路：GRACE/GRACE-FO 重力 → 陆地水储量同化 → 地下水储量 gws_inst（mm）
任务：severity none / depleted；detected 与 severity 一致；窗口取 gws 的逐像元百分位。
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
RAW = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "grace_gw")
OUT = os.path.join(ROOT, "datasets", "04_passive_microwave", "waterlogging_grace")
IMG = os.path.join(OUT, "images")

WIN = 16
STRIDE = 8
MIN_VALID = 0.5
POS_PER_FILE = 6
NEG_PER_FILE = 6
SEED = 42
OUT_SIZE = 384
LATB = (18.0, 54.0)
LONB = (73.0, 135.0)

PROMPT = ("This is a GRACE-derived surface soil-moisture percentile map (0-100) over China: high "
          "values mean the surface soil is unusually wet. Output a JSON object with fields: "
          "disaster_type, detected (true/false), source, physical_quantity, severity, criterion. "
          "severity must be one of these exact words: none / waterlogging.")


def load_gws(fp):
    ds = Dataset(fp)
    g = np.array(ds.variables["sfsm_inst"][0], dtype="float32")
    lat = np.array(ds.variables["lat"][:], dtype="float64")
    lon = np.array(ds.variables["lon"][:], dtype="float64")
    ds.close()
    g[(g < 0) | (g > 5000)] = np.nan
    return g, lat, lon


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="BrBG", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("surface soil moisture pct", fontsize=6.5)
    fig.text(0.04, 0.075, "GRACE-DA weekly, 0.25 deg", fontsize=6)
    fig.text(0.04, 0.035, "brown = dry, blue = saturated", fontsize=6, color="#666")
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

    files = sorted(glob.glob(os.path.join(RAW, "*.nc4")))
    if a.limit:
        files = files[:a.limit]
    print("文件 %d" % len(files))

    stack, lat0, lon0, stamps = [], None, None, []
    for fp in files:
        g, lat, lon = load_gws(fp)
        m = re.search(r"\.A(\d{8})\.", os.path.basename(fp))
        stack.append(g)
        if lat0 is None:
            lat0, lon0 = lat, lon
        stamps.append(m.group(1) if m else "unknown")
    stack = np.stack(stack)
    print("堆叠", stack.shape)

    pct = np.full_like(stack, np.nan)
    D = stack.shape[0]
    for i in range(D):
        v = stack[i]
        lower = np.nansum(stack < v, axis=0)
        valid = np.sum(np.isfinite(stack), axis=0)
        pct[i] = np.where(valid >= 10, 100.0 * lower / np.maximum(valid, 1), np.nan)

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
                elif mp <= 30:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for det, (r0, c0, la, lo, mp) in [(True, x) for x in pos[:POS_PER_FILE]] + \
                [(False, x) for x in neg[:NEG_PER_FILE]]:
            sev = "waterlogging" if det else "none"
            nm = "wlg_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            render(p[r0:r0 + WIN, c0:c0 + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "waterlogging",
                "detected": bool(det),
                "source": "GRACE",
                "physical_quantity": "soil_moisture_percentile",
                "severity": sev,
                "criterion": ("surface soil moisture percentile >= 90 = waterlogging risk "
                              "(GRACE-DA weekly, 2020-2022 percentile)"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if i % 20 == 0:
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
        n = min(max(2, int(round(len(lst) * 0.2))), 80, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])
    json.dump(train, open(os.path.join(OUT, "train_waterlogging_grace_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_waterlogging_grace_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
