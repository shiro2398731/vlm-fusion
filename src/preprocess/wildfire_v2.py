# -*- coding: utf-8 -*-
r"""
野火/林火（源 3 热红外）—— 双产品重做版

判据（FIRMS VIIRS 火点）：
  过滤 type=2（静态地表热源：工厂/冶炼/地热，非野火）
  过滤 confidence='l'（低置信度）
  窗内有效火点数 >= 5  →  fire
  否则                 →  no_fire

双产品：
  本征场 = 中红外亮温 bright_ti4 场（K）—— 热红外源直接观测的辐射量
  特色场 = FRP 累积场（MW）—— 由普朗克反演的火强度，火灾专属

数据：data\raw\03_thermal_infrared\firms\chn\fire_YYYY-MM-DD.csv（366 天）

用法: python wildfire_v2.py [--grid 5] [--max-days N]
"""
import os
import re
import sys
import glob
import json
import csv
import argparse
import warnings
from datetime import date
import numpy as np
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RAW = r"C:\vlmdisaster\data\raw\03_thermal_infrared\firms\chn"
OUT = r"C:\vlmdisaster\datasets\03_thermal_infrared\wildfire"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_DAY = 2
MIN_PTS = 5

# 中国范围
W, S, E, N = 73.0, 18.0, 135.0, 54.0

PROMPT = ("This is a satellite thermal-infrared fire detection map over a region in China. "
          "Classify: no_fire or fire.")


def load_pts(fp):
    """读一天火点，返回 [(lat, lon, ti4, frp)]"""
    out = []
    with open(fp, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                t = int(row.get("type", "0") or 0)
                conf = (row.get("confidence") or "").strip().lower()
                if t == 2:
                    continue                      # 静态热源，非野火
                if conf == "l":
                    continue                      # 低置信度
                lat = float(row["latitude"]); lon = float(row["longitude"])
                ti4 = float(row["bright_ti4"]); frp = float(row["frp"] or 0)
                out.append((lat, lon, ti4, frp))
            except Exception:
                continue
    return out


def grid_of(pts, la, lo, g, n=40):
    """把点在 g 度窗内累加成 n×n 网格"""
    ti4 = np.zeros((n, n), dtype=float)
    frp = np.zeros((n, n), dtype=float)
    cnt = np.zeros((n, n), dtype=int)
    for lat, lon, t, f_ in pts:
        if not (la <= lat < la + g and lo <= lon < lo + g):
            continue
        ix = min(n - 1, int((lon - lo) / g * n))
        iy = min(n - 1, int((lat - la) / g * n))
        ti4[iy, ix] += t
        frp[iy, ix] += f_
        cnt[iy, ix] += 1
    with np.errstate(invalid="ignore"):
        ti4m = np.where(cnt > 0, ti4 / np.maximum(cnt, 1), np.nan)
    return ti4m, frp, cnt


def draw(field, cnt, la, lo, g, path, lab, kind):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=96)
    if kind == "ti4":
        im = ax.imshow(field, origin="lower", extent=[lo, lo + g, la, la + g],
                       cmap="inferno", vmin=300, vmax=420)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("brightness temp (K)", fontsize=7)
    else:
        im = ax.imshow(field, origin="lower", extent=[lo, lo + g, la, la + g],
                       cmap="YlOrRd", vmin=0, vmax=max(50, float(np.nanmax(field))))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("FRP (MW)", fontsize=7)
    if int(cnt.sum()) == 0:
        ax.text(0.5, 0.5, "No fire hotspot", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
    ax.set_xlabel("Lon", fontsize=7); ax.set_ylabel("Lat", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("%s" % lab, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=5.0)
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    for f in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(f)

    fs = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    if args.max_days:
        fs = fs[:args.max_days]
    print("天数 %d | 窗 %.1f°" % (len(fs), args.grid))
    if not fs:
        return

    g = args.grid
    lats = np.arange(S, N - g + 1e-9, g)
    lons = np.arange(W, E - g + 1e-9, g)
    recs, cnt = [], {}
    for i, f in enumerate(fs, 1):
        stamp = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        stamp = stamp.group(1) if stamp else str(i)
        pts = load_pts(f)
        by = {}
        for la in lats:
            for lo in lons:
                ti4m, frp, c = pts_in(pts, la, lo, g)
                n = int(c.sum())
                lab = "fire" if n >= MIN_PTS else "no_fire"
                # 选代表性窗：fire 挑点数多的，no_fire 挑空窗
                score = n if lab == "fire" else 0
                by.setdefault(lab, []).append((score, la, lo, ti4m, frp, c))
        picked = []
        for lab, lst in by.items():
            lst.sort(key=lambda t: -t[0])
            picked += [(lab, t) for t in lst[:MAX_PER_DAY]]

        for lab, (_, la, lo, ti4m, frp, c) in picked:
            cnt[lab] = cnt.get(lab, 0) + 1
            tag = "%s%d%s%d" % ("N" if la >= 0 else "S", abs(int(la)),
                                "E" if lo >= 0 else "W", abs(int(lo)))
            for sfx, kind, prod in (("_ti4", "ti4", "intrinsic_brightness_temp"),
                                    ("_frp", "frp", "feature_frp")):
                png = os.path.join(IMG, "%s_%s_%s%s.png" % (stamp, tag, lab, sfx))
                draw(ti4m, c, la, lo, g, png, "%s %s" % (stamp, lab), kind)
                ans = ("Active wildfire detected: %d hotspot pixels with FRP." % int(c.sum())
                       if lab == "fire" else
                       "No significant wildfire activity in this region.")
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ans}],
                    "images": [png], "product": prod, "label": lab,
                    "hotspots": int(c.sum()), "stamp": stamp})
        if i % 30 == 0:
            print("  %d/%d %s" % (i, len(fs), dict(cnt)), flush=True)

    import random, collections
    random.seed(3)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_wildfire.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_wildfire_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n图 %d | train %d | val %d" % (len(recs), len(train), len(val)))
    print("分布:", dict(cnt))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


def pts_in(pts, la, lo, g, n=40):
    ti4 = np.zeros((n, n)); frp = np.zeros((n, n)); cnt = np.zeros((n, n), dtype=int)
    for lat, lon, t, f_ in pts:
        if not (la <= lat < la + g and lo <= lon < lo + g):
            continue
        ix = min(n - 1, int((lon - lo) / g * n))
        iy = min(n - 1, int((lat - la) / g * n))
        ti4[iy, ix] += t; frp[iy, ix] += f_; cnt[iy, ix] += 1
    with np.errstate(invalid="ignore"):
        m = np.where(cnt > 0, ti4 / np.maximum(cnt, 1), np.nan)
    return m, frp, cnt


if __name__ == "__main__":
    main()
