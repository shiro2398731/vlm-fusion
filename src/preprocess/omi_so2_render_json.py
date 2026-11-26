# -*- coding: utf-8 -*-
r"""
OMI SO2 切窗 + 结构化 JSON（新元素：volcano_so2）

物理链路
--------
  OMI/Aura 紫外光谱仪 → 差分吸收光谱（DOAS）反演 SO2 垂直柱浓度（DU）
  → 0.25° 日网格（OMSO2e）
真值
----
  MSVOLSO2L4 多卫星火山 SO2 数据库：哪一天、哪座火山在排放 SO2
  → 正样本 = 事件日、以火山为中心的窗口；负样本 = 远离火山的背景窗

任务设计（防塌缩）
------------------
  detected：窗口是否在火山事件附近（真值）
  数值字段：窗口最大 SO2 柱浓度 max_so2_du（DU）——只能从图上读
  severity：按 max_so2_du 分档（<1 none / 1-5 light / 5-20 moderate / >=20 heavy，本文设定）

用法
----
  .\venv\Scripts\python.exe src\preprocess\omi_so2_render_json.py --top-days 3   # 先验
  .\venv\Scripts\python.exe src\preprocess\omi_so2_render_json.py --top-days 120 # 全量
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
RAW = os.path.join(ROOT, "data", "raw", "11_atmospheric", "omi_so2")
DB = os.path.join(ROOT, "data", "raw", "11_atmospheric", "volso2_l4",
                  "MSVOLSO2L4_v04-00-2026m0513.txt")
OUT = os.path.join(ROOT, "datasets", "11_atmospheric", "volcano_so2")
IMG = os.path.join(OUT, "images")

WIN = 128              # 0.25° × 128 = 32°，覆盖下风向羽流
STRIDE = 64
MIN_VALID = 0.3
POS_PER_DAY = 4
NEG_PER_DAY = 4
SEED = 42
OUT_SIZE = 384
SO2_MIN = 0.5          # 正样本窗口至少要有 0.5 DU 的 SO2（弱羽流也能判）

PROMPT = ("This is an OMI satellite SO2 vertical column density map (Dobson Units, DU) "
          "over a region near an active volcano. Output a JSON object with fields: "
          "disaster_type, detected (true/false), source, physical_quantity, severity, "
          "max_so2_du, criterion. severity must be one of these exact words: "
          "none / light / moderate / heavy.")


def load_db():
    """返回 日期(YYYYMMDD) -> [(lat, lon, mass_kt)]，只留 2020-2024"""
    ev = collections.defaultdict(list)
    for line in open(DB, encoding="utf-8", errors="replace"):
        p = line.strip().split("\t")
        if len(p) < 7:
            continue
        try:
            y, m, d = int(p[4]), int(p[5]), int(p[6])
        except ValueError:
            continue
        if not (2020 <= y <= 2024):
            continue
        try:
            lat, lon = float(p[1]), float(p[2])
        except ValueError:
            continue
        try:
            mass = float(p[10]) if len(p) > 10 else -999
        except ValueError:
            mass = -999
        ev["%04d%02d%02d" % (y, m, d)].append((lat, lon, mass))
    return ev


def load(fp):
    ds = Dataset(fp)
    g = ds.groups["HDFEOS"].groups["GRIDS"].groups["OMI Total Column Amount SO2"].groups["Data Fields"]
    so2 = np.ma.filled(np.array(g.variables["ColumnAmountSO2"][:], dtype="float64"), np.nan)
    qa = np.array(g.variables["QualityFlags_SO2"][:], dtype="float64")
    sza = np.ma.filled(np.array(g.variables["SolarZenithAngle"][:], dtype="float64"), np.nan)
    lat = np.array(g.variables["Latitude"][:], dtype="float64")
    lon = np.array(g.variables["Longitude"][:], dtype="float64")
    ds.close()
    ok = np.isfinite(so2) & (so2 > -10) & (so2 < 2000) & np.isfinite(sza) & (sza < 70)
    so2 = np.where(ok, so2, np.nan)
    return so2, lat, lon


def severity_of(mx):
    if mx < 1:
        return "none"
    if mx < 5:
        return "light"
    if mx < 20:
        return "moderate"
    return "heavy"


def render(win, path, stamp, la, lo, mx):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="plasma", vmin=0, vmax=30.0, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("SO2 column (DU)", fontsize=7)
    fig.text(0.04, 0.075, "OMI/Aura OMSO2e, daily 0.25 deg", fontsize=6)
    fig.text(0.04, 0.035, "white = no retrieval / cloud", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-days", type=int, default=120,
                    help="按事件总 SO2 质量取前 N 个事件日（0=全部）")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    ev = load_db()
    print("事件库：2020-2024 共 %d 个事件日" % len(ev))

    files = {}
    for f in glob.glob(os.path.join(RAW, "*.he5")):
        m = re.search(r"(\d{4})m(\d{2})(\d{2})", os.path.basename(f))
        if m:
            files[m.group(1) + m.group(2) + m.group(3)] = f

    days = [d for d in ev if d in files]
    # 按当天事件总质量排序，取前 N 天（质量 -999 视为 0）
    days.sort(key=lambda d: -sum(max(0, x[2]) for x in ev[d]))
    if args.top_days:
        days = days[:args.top_days]
    print("可渲染事件日 %d 个" % len(days))

    records = []
    random.seed(SEED)
    for fi, d in enumerate(days, 1):
        so2, lat, lon = load(files[d])
        h, w = so2.shape
        # 事件位置 -> 窗口
        pos_windows = {}
        for (la, lo, _m) in ev[d]:
            r0 = int(np.clip((la + 90) / 180 * h, 0, h - WIN))
            c0 = int(np.clip((lo + 180) / 360 * w, 0, w - WIN))
            # 事件窗口直接以火山为中心，不对齐步长
            pos_windows[(r0, c0)] = (la, lo)
        negatives = []
        for _ in range(200):
            if len(negatives) >= NEG_PER_DAY:
                break
            r0 = random.randrange(0, h - WIN + 1, STRIDE)
            c0 = random.randrange(0, w - WIN + 1, STRIDE)
            win = so2[r0:r0 + WIN, c0:c0 + WIN]
            fin = np.isfinite(win)
            if fin.mean() < MIN_VALID:
                continue
            mx = float(np.nanmax(win))
            if mx >= SO2_MIN:
                continue
            la = float(np.nanmean(lat[r0:r0 + WIN, c0:c0 + WIN]))
            lo = float(np.nanmean(lon[r0:r0 + WIN, c0:c0 + WIN]))
            negatives.append((r0, c0, la, lo, mx))

        picks = []
        # 在火山周围 3x3 个候选窗口里找羽流（羽流常在下风向，不一定正对火山口）
        cand = {}
        for (r0, c0), (la0, lo0) in pos_windows.items():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(np.clip(r0 + dr * WIN // 2, 0, h - WIN))
                    cc = int(np.clip(c0 + dc * WIN // 2, 0, w - WIN))
                    win = so2[rr:rr + WIN, cc:cc + WIN]
                    if np.isfinite(win).mean() < MIN_VALID:
                        continue
                    mx = float(np.nanmax(win))
                    if mx < SO2_MIN:
                        continue
                    la = float(np.nanmean(lat[rr:rr + WIN, cc:cc + WIN]))
                    lo = float(np.nanmean(lon[rr:rr + WIN, cc:cc + WIN]))
                    cand[(rr, cc)] = (mx, rr, cc, la, lo)
        cand = sorted(cand.values(), key=lambda t: -t[0])
        for mx, rr, cc, la, lo in cand[:POS_PER_DAY]:
            picks.append((True, rr, cc, la, lo, mx))
        for (r0, c0, la, lo, mx) in negatives:
            picks.append((False, r0, c0, la, lo, mx))

        for det, r0, c0, la, lo, mx in picks:
            sev = severity_of(mx)
            name = "so2_%s_r%d_c%d.png" % (d, r0, c0)
            p = os.path.join(IMG, name)
            render(so2[r0:r0 + WIN, c0:c0 + WIN], p, d, la, lo, mx)
            ans = {
                "disaster_type": "volcano_so2",
                "detected": bool(det),
                "source": "ATM",
                "physical_quantity": "so2_column",
                "severity": sev,
                "max_so2_du": round(mx, 2),
                "criterion": ("volcanic SO2 event from MSVOLSO2L4; SO2 column in DU; "
                              "severity bands (<1/1-5/5-20/>=20 DU) are study-defined"),
            }
            records.append({
                "image": p, "label": sev, "stamp": d,
                "max_so2_du": round(mx, 2),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if fi % 20 == 0 or fi == len(days):
            print("  [%d/%d] %s  累计 %d" % (fi, len(days), d, len(records)))

    # 去重（同一张图只保留一条，避免训练/验证集重叠）
    seen = set()
    uniq = []
    for r in records:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        uniq.append(r)
    records = uniq

    random.seed(SEED)
    by = collections.defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for k in sorted(by):
        lst = by[k][:]
        random.shuffle(lst)
        n = min(max(2, int(round(len(lst) * 0.2))), 30, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_volcano_so2_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_volcano_so2_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()
