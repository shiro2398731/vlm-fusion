# -*- coding: utf-8 -*-
r"""
AMSR2 海冰密集度 · 切窗 + 结构化 JSON（新元素：sea_ice_amsr2）

与旧版 seaice_amsr2_render.py 的区别
------------------------------------
旧版：每天一张全半球图，答案写成自然语言
本版：把半球网格切成 128x128 像素窗口，每窗一个样本；答案写成结构化 JSON，
      带窗口内「冰覆盖比例 %」和「平均密集度 %」——这是防止模型塌缩的读数证据。

物理链路（论文引用）
------------------------
  AMSR2 被动微波辐射计（18.7/36.5/89 GHz 亮温 T_b）
    → NASA Team / ASI 算法反演 海冰密集度（%）
    → 窗口内「密集度 >= 15% 的像元占比」= 冰覆盖比例（%）
  判据：WMO / OSI SAF 定义「密集度 >= 15% 为海冰区」，本图在窗口尺度上使用；
        severity 四档（none/light/moderate/heavy）为本文设定，论文需注明。

输出
----
  datasets/04_passive_microwave/sea_ice_amsr2/images/*.png
  datasets/04_passive_microwave/sea_ice_amsr2/train_sea_ice_amsr2_json.json
  datasets/04_passive_microwave/sea_ice_amsr2/train_sea_ice_amsr2_json_val.json

用法
----
  .\venv\Scripts\python.exe src\preprocess\seaice_amsr2_render_json.py --limit-days 2   # 先验
  .\venv\Scripts\python.exe src\preprocess\seaice_amsr2_render_json.py                 # 全量
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

PILImage.MAX_PIXELS = None

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "seaice_amsr2")
OUT = os.path.join(ROOT, "datasets", "04_passive_microwave", "sea_ice_amsr2")
IMG = os.path.join(OUT, "images")

ICE = 15.0            # WMO：密集度 >= 15% 为海冰区
WIN = 128             # 窗口像素
STRIDE = 128
MIN_VALID = 0.5       # 窗内有效像元（非陆地/缺失）比例下限
MAX_PER_CLASS = 8     # 每天每类最多取几个窗（平衡用）
OUT_SIZE = 384
SEED = 42

PROMPT = ("This is an AMSR2 passive-microwave sea-ice concentration window (percent) "
          "of the Northern Hemisphere. Output a JSON object with fields: "
          "disaster_type, detected (true/false), source, physical_quantity, severity, "
          "ice_frac_pct, mean_conc_pct, criterion. severity must be one of these exact "
          "words: none / light / moderate / heavy.")

SEV_BANDS = [(15.0, "light"), (50.0, "moderate"), (85.0, "heavy")]   # 本文设定


def severity_of(frac):
    if frac < ICE:
        return "none"
    for lo, name in reversed(SEV_BANDS):
        if frac >= lo:
            return name
    return "light"


def load(fp):
    ds = Dataset(fp)
    g = ds.groups["HDFEOS"].groups["GRIDS"].groups["NpPolarGrid12km"]
    df = g.groups["Data Fields"]
    ice = np.ma.filled(np.array(df.variables["SI_12km_NH_ICECON_DAY"][:], dtype="float64"), np.nan)
    lat = np.ma.filled(np.array(g.variables["lat"][:], dtype="float64"), np.nan)
    lon = np.ma.filled(np.array(g.variables["lon"][:], dtype="float64"), np.nan)
    ds.close()
    ice[(ice == 110) | (ice == 120)] = np.nan     # 缺失 / 陆地
    ice[(ice < 0) | (ice > 100)] = np.nan
    return ice, lat, lon


def render(win, path, stamp, r0, c0, latc, lonc):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="Blues", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  window r%d c%d  (%.1fN %.1fE)" % (stamp, r0, c0, latc, lonc), fontsize=7)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ice concentration (%)", fontsize=7)
    fig.text(0.04, 0.075, "AMSR2 12.5 km  AU_SI12  (passive microwave)", fontsize=6)
    fig.text(0.04, 0.035, "WMO: concentration >= 15% marks the ice zone", fontsize=6, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=0, help="只处理前 N 个文件，先验用")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.he5")))
    if args.limit_days:
        files = files[:args.limit_days]
    print("待渲染文件:", len(files))
    if not files:
        return

    pos_recs, neg_recs = [], []
    per_day = collections.Counter()

    for fi, fp in enumerate(files, 1):
        m = re.search(r"(\d{8})", os.path.basename(fp))
        if not m:
            continue
        stamp = m.group(1)
        try:
            ice, lat, lon = load(fp)
        except Exception as e:
            print("  %s 读取失败 %s" % (stamp, str(e)[:70]))
            continue

        h, w = ice.shape
        day_pos, day_neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                wi = ice[r0:r0 + WIN, c0:c0 + WIN]
                la = lat[r0:r0 + WIN, c0:c0 + WIN]
                lo = lon[r0:r0 + WIN, c0:c0 + WIN]
                fin = np.isfinite(wi)
                if fin.mean() < MIN_VALID or fin.sum() < 1000:
                    continue
                v = wi[fin]
                ice_frac = float(np.mean(v >= ICE) * 100.0)
                mean_conc = float(np.mean(v))
                latc = float(np.nanmean(la[fin]))
                lonc = float(np.nanmean(lo[fin]))
                det = ice_frac >= ICE
                rec = {"r0": r0, "c0": c0, "win": wi, "ice_frac": ice_frac,
                       "mean_conc": mean_conc, "lat": latc, "lon": lonc, "stamp": stamp}
                (day_pos if det else day_neg).append(rec)

        random.seed(SEED + fi)
        random.shuffle(day_pos)
        random.shuffle(day_neg)
        for rec in day_pos[:MAX_PER_CLASS]:
            pos_recs.append(rec)
        for rec in day_neg[:MAX_PER_CLASS]:
            neg_recs.append(rec)
        per_day["pos"] += min(len(day_pos), MAX_PER_CLASS)
        per_day["neg"] += min(len(day_neg), MAX_PER_CLASS)
        if fi % 10 == 0 or fi == len(files):
            print("  [%d/%d] %s  pos+%d neg+%d (累计 %d/%d)"
                  % (fi, len(files), stamp, min(len(day_pos), MAX_PER_CLASS),
                     min(len(day_neg), MAX_PER_CLASS), len(pos_recs), len(neg_recs)))

    # ---- 渲染 + 结构化 JSON ----
    records = []
    for rec in pos_recs + neg_recs:
        stamp, r0, c0 = rec["stamp"], rec["r0"], rec["c0"]
        sev = severity_of(rec["ice_frac"])
        name = "seaice_%s_r%d_c%d.png" % (stamp, r0, c0)
        p = os.path.join(IMG, name)
        render(rec["win"], p, stamp, r0, c0, rec["lat"], rec["lon"])
        answer = {
            "disaster_type": "sea_ice",
            "detected": bool(rec["ice_frac"] >= ICE),
            "source": "MW",
            "physical_quantity": "sea_ice_concentration",
            "severity": sev,
            "ice_frac_pct": round(rec["ice_frac"], 2),
            "mean_conc_pct": round(rec["mean_conc"], 2),
            "criterion": ("ice zone defined by WMO/OSI SAF 15% concentration threshold; "
                          "severity bands (none<15, light 15-50, moderate 50-85, heavy>=85) "
                          "are study-defined"),
        }
        records.append({
            "image": p,
            "label": sev,
            "stamp": stamp,
            "window": [r0, c0],
            "ice_frac_pct": round(rec["ice_frac"], 2),
            "mean_conc_pct": round(rec["mean_conc"], 2),
            "conversations": [
                {"from": "human", "value": "<image>" + PROMPT},
                {"from": "gpt", "value": json.dumps(answer, ensure_ascii=False)},
            ],
        })

    # ---- 分层划分 ----
    random.seed(SEED)
    by = collections.defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for k in sorted(by):
        lst = by[k][:]
        random.shuffle(lst)
        n = min(max(10, int(round(len(lst) * 0.2))), len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_sea_ice_amsr2_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_sea_ice_amsr2_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("按天采样:", dict(per_day))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})

    frac = [r["ice_frac_pct"] for r in records]
    print("ice_frac: min %.1f  max %.1f  中位 %.1f" % (min(frac), max(frac), float(np.median(frac))))
    print("输出:", OUT)


if __name__ == "__main__":
    main()
