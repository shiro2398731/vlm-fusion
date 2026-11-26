# -*- coding: utf-8 -*-
r"""
渲染 GPM IMERG 降水图（服务"洪水/台风/龙卷风"元素，源 6 雷达降水）。

输入：data\raw\06_radar_precipitation\gpm\3B-HHR-L*.HDF5（半小时，一天 48 个）
输出：datasets\06_radar_precipitation\<element>\images\*.png
      datasets\06_radar_precipitation\<element>\train_gpm.json

逻辑：
  1. 把同一天的所有半小时片合成"当日累积降水"（mm）
  2. 按地理块切窗，每窗渲一张降水图
  3. 标签：窗内累积降水强度分级（dry / moderate_rain / heavy_rain）

元素映射：GPM 数据同时服务 洪水/洪涝、飓风/台风、龙卷风——统一按"强降水"渲染，
         三个元素共用同一批图（标签按降水强度，不细分灾害类型）。
"""
import os
import re
import sys
import glob
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = r"C:\vlmdisaster\data\raw\06_radar_precipitation\gpm"
OUT = r"C:\vlmdisaster\datasets\06_radar_precipitation"
ELEMENT = "flood"          # 输出到 06_radar_precipitation/flood
IMG_DIR = os.path.join(OUT, ELEMENT, "images")
OUT_JSON = os.path.join(OUT, ELEMENT, "train_gpm.json")

WIN = 256          # 窗口边长（0.1°格网 → 256 像素 ≈ 25.6° 太大；用经纬度窗口）
HALF = 12          # 以某中心点切 ±12° 的窗口（≈240像素）
STRIDE_DEG = 16
MIN_RAIN_MM = 0.5  # 窗内均值超过这个才算"有雨"

GRADE = [
    ("dry", 0.0, 0.5),
    ("moderate_rain", 0.5, 5.0),
    ("heavy_rain", 5.0, 1e9),
]

# 采样策略：只切"有显著降水的区域"，避免全球均匀切出几千张无雨的海面图。
LAND_FRAC_MIN = 0.5      # 窗内降雨像素占比下限
RAIN_MEAN_MIN = 0.3      # 窗内平均降水下限（mm）
MAX_PER_DAY = 150        # 每天最多产出窗口数（避免爆炸）


def grade_of(mm):
    for name, lo, hi in GRADE:
        if lo <= mm < hi:
            return name
    return GRADE[-1][0]


def load_day(date_str):
    fs = sorted(glob.glob(os.path.join(RAW, f"*{date_str}*.HDF5")))
    if not fs:
        return None, None, None
    with h5py.File(fs[0], "r") as h:
        lat = h["Grid"]["lat"][:]
        lon = h["Grid"]["lon"][:]
    total = None
    for f in fs:
        with h5py.File(f, "r") as h:
            a = h["Grid"]["precipitation"][0].astype("float64")
            a[a < -1000] = 0.0
            a = a * 0.5   # mm/hr * 0.5h
        total = a if total is None else total + a
    return total, lat, lon


def render(sub, path, date, mm):
    vmax = max(1.0, float(np.percentile(sub, 99)))
    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    m = ax.pcolormesh(sub, cmap="Blues", vmin=0, vmax=vmax, shading="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"GPM precip {date} | {mm:.1f}mm", fontsize=8)
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04).set_label("mm/day", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((384, 384), PILImage.BILINEAR).save(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    # 按日期分组文件
    dates = sorted(set(re.search(r"3IMERG\.(\d{8})-", os.path.basename(f)).group(1)
                       for f in glob.glob(os.path.join(RAW, "*.HDF5"))))
    print(f"GPM 日期: {dates}")
    records = []
    for date in dates:
        total, lat, lon = load_day(date)
        if total is None:
            print(f"  · {date}: 无文件")
            continue
        H, W = total.shape
        # 先找出所有"有雨"的窗口，再按降水总量排序取前 MAX_PER_DAY 个
        cands = []
        dry_cands = []
        for ly in range(0, H - 2 * HALF, STRIDE_DEG):
            for lx in range(0, W - 2 * HALF, STRIDE_DEG):
                sub = total[ly:ly + 2 * HALF, lx:lx + 2 * HALF]
                mean_mm = float(sub.mean())
                if mean_mm >= RAIN_MEAN_MIN:
                    cands.append((mean_mm, ly, lx, sub))
                else:
                    dry_cands.append((mean_mm, ly, lx, sub))
        # 有雨窗：按降水降序取前 MAX_PER_DAY，并适量保留弱雨区
        cands.sort(key=lambda x: -x[0])
        picked = cands[:MAX_PER_DAY]
        weak = [c for c in cands if c[0] < 5.0][:MAX_PER_DAY // 2]
        # 无雨（dry）窗：均匀抽取，数量约取有雨窗的一半，做二分类负样本
        import random as _rnd
        _rnd.seed(42)
        dry_pick = _rnd.sample(dry_cands, min(len(dry_cands), MAX_PER_DAY // 2)) if dry_cands else []
        seen = set()
        final = []
        for c in picked + weak + dry_pick:
            key = (c[1], c[2])
            if key not in seen:
                seen.add(key)
                final.append(c)
        for mean_mm, ly, lx, sub in final:
            lab = grade_of(mean_mm)
            fn = f"gpm_{date}_{ly}_{lx}.png"
            render(sub, os.path.join(IMG_DIR, fn), date, mean_mm)
            records.append({
                "image": fn, "date": date, "label": lab,
                "mean_mm": round(mean_mm, 2), "max_mm": round(float(sub.max()), 1),
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a GPM satellite precipitation accumulation map "
                              "over a region. Classify the rainfall intensity: dry, moderate rain, or heavy rain."},
                    {"from": "gpt",
                     "value": {
                         "dry": "This region is dry, with little to no rainfall.",
                         "moderate_rain": "This region has moderate rainfall.",
                         "heavy_rain": "This region has heavy rainfall, likely causing flooding.",
                     }[lab]},
                ],
            })
        print(f"  ✓ {date}: 有雨窗 {len(cands)} 个、无雨窗 {len(dry_cands)} 个 → 取 {len(final)} 个")

    json.dump(records, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter, defaultdict
    import random as _rnd
    _rnd.seed(11)
    print(f"\n合计 {len(records)} 样本")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    # 划验证集（按类分层）
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        _rnd.shuffle(lst)
        k = min(30, len(lst) // 5)
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)} ({dict(Counter(r['label'] for r in val))})")


if __name__ == "__main__":
    main()
