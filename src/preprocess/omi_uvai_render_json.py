# -*- coding: utf-8 -*-
r"""
OMI 紫外气溶胶指数（UVAI）切窗 + 结构化 JSON（新元素：uvai_dust）

物理链路
--------
  OMI/Aura 紫外光谱仪（354/388 nm）→ 吸收性气溶胶指数 UVAI（无量纲）
  UVAI > 1 表示吸收性气溶胶（沙尘、烟羽）；< 0 表示散射性气溶胶/云
真值/判据
---------
  判据：UVAI >= 1.0 为吸收性气溶胶（文献常用阈值，Hsu et al. 1999）
  与 sanddust（MODIS AOD）是**不同源、不同物理量**的同灾害补源

任务设计（防塌缩）
------------------
  detected：窗内最大 UVAI 是否 >= 1.0（读数 + 阈值）
  数值字段：max_uvai（只能从图上读）
  severity：none<1 / light 1-2 / moderate 2-4 / heavy>=4（本文设定）

用法
----
  .\venv\Scripts\python.exe src\preprocess\omi_uvai_render_json.py --top-days 5
  .\venv\Scripts\python.exe src\preprocess\omi_uvai_render_json.py --top-days 120
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
RAW = os.path.join(ROOT, "data", "raw", "11_atmospheric", "omi_uvai")
OUT = os.path.join(ROOT, "datasets", "11_atmospheric", "uvai_dust")
IMG = os.path.join(OUT, "images")

WIN = 8                # 1° × 8 = 8°
STRIDE = 8
MIN_VALID = 0.5
POS_PER_DAY = 4
NEG_PER_DAY = 4
UVAI_MIN = 1.0         # 吸收性气溶胶阈值
SEED = 42
OUT_SIZE = 384

PROMPT = ("This is an OMI UV Aerosol Index (UVAI) map, dimensionless. UVAI > 1 indicates "
          "absorbing aerosols such as dust or smoke; negative values indicate scattering "
          "aerosols/clouds. Output a JSON object with fields: disaster_type, detected "
          "(true/false), source, physical_quantity, severity, max_uvai, criterion. "
          "severity must be one of these exact words: none / light / moderate / heavy.")


def severity_of(v):
    if v < 1:
        return "none"
    if v < 2:
        return "light"
    if v < 4:
        return "moderate"
    return "heavy"


def load(fp):
    ds = Dataset(fp)
    g = ds.groups["HDFEOS"].groups["GRIDS"].groups["Aerosol NearUV Grid"]
    v = g.groups["Data Fields"].variables["UVAerosolIndex"]
    a = np.ma.filled(np.array(v[:], dtype="float64"), np.nan)
    ds.close()
    a[(a < -100) | (a > 100)] = np.nan
    return a


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="YlOrRd", vmin=-1, vmax=5, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("UVAI (dimensionless)", fontsize=7)
    fig.text(0.04, 0.075, "OMI/Aura OMAERUVd, daily 1 deg", fontsize=6)
    fig.text(0.04, 0.035, "UVAI > 1 = absorbing aerosol (dust/smoke)", fontsize=6, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-days", type=int, default=120)
    ap.add_argument("--max-files", type=int, default=0, help="先验用：只扫前 N 个文件")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = {}
    for f in glob.glob(os.path.join(RAW, "*.he5")):
        m = re.search(r"(\d{4})m(\d{2})(\d{2})", os.path.basename(f))
        if m:
            files[m.group(1) + m.group(2) + m.group(3)] = f
    if args.max_files:
        files = dict(sorted(files.items())[:args.max_files])
    print("UVAI 日文件 %d 个" % len(files))

    # 第一遍：算每天最大值（挑强沙尘日）
    day_max = {}
    for d, fp in files.items():
        try:
            a = load(fp)
        except Exception:
            continue
        mx = float(np.nanmax(a)) if np.isfinite(a).any() else -999
        day_max[d] = mx
    days = sorted([d for d, mx in day_max.items() if mx >= 2.0], key=lambda d: -day_max[d])
    if args.top_days:
        days = days[:args.top_days]
    print("候选强事件日 %d 个（取 %d）" % (len([d for d, m in day_max.items() if m >= 2.0]), len(days)))
    if not days:
        print("没有可用事件日")
        return

    records = []
    random.seed(SEED)
    for fi, d in enumerate(days, 1):
        a = load(files[d])
        h, w = a.shape
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                win = a[r0:r0 + WIN, c0:c0 + WIN]
                fin = np.isfinite(win)
                if fin.mean() < MIN_VALID:
                    continue
                mx = float(np.nanmax(win))
                la = 89.5 - (r0 + WIN / 2)
                lo = -179.5 + (c0 + WIN / 2)
                item = (r0, c0, la, lo, mx, float(np.nanmean(win)))
                if mx >= UVAI_MIN:
                    pos.append(item)
                elif mx < 0.3:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for (r0, c0, la, lo, mx, mean_v) in pos[:POS_PER_DAY] + neg[:NEG_PER_DAY]:
            sev = severity_of(mx)
            det = mx >= UVAI_MIN
            name = "uvai_%s_r%d_c%d.png" % (d, r0, c0)
            p = os.path.join(IMG, name)
            render(a[r0:r0 + WIN, c0:c0 + WIN], p, d, la, lo)
            ans = {
                "disaster_type": "dust_aerosol",
                "detected": bool(det),
                "source": "ATM",
                "physical_quantity": "uv_aerosol_index",
                "severity": sev,
                "max_uvai": round(mx, 2),
                "mean_uvai": round(mean_v, 2),
                "criterion": ("UVAI >= 1.0 indicates absorbing aerosol (dust/smoke), "
                              "Hsu et al. 1999; severity bands study-defined"),
            }
            records.append({
                "image": p, "label": sev, "stamp": d,
                "max_uvai": round(mx, 2),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if fi % 20 == 0 or fi == len(days):
            print("  [%d/%d] %s  累计 %d" % (fi, len(days), d, len(records)))

    # 去重 + 分层划分
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

    json.dump(train, open(os.path.join(OUT, "train_uvai_dust_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_uvai_dust_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
