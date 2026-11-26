# -*- coding: utf-8 -*-
r"""
渲染 MODIS LST 干旱样本（服务热红外 TIR 的"干旱"元素）— v2 判据重做版。

v1 问题（已废弃）：判据用窗口内白天"最高点" dmax>35 且夜间均值>20，
  128×128 窗口里只要有一小块热斑（城市热岛/裸岩/沙地边缘）整格就判 drought，
  导致正样本被稀释到 1379/14192（1:9），且"干旱"这个持续概念被单期极值代替。

v2 判据（本版）：从"有没有一个极端点"改为"这块区域主体是不是高温"。
  drought : 白天 P90 > 35°C  且  夜间 P50 > 20°C  且  白天高温像元占比 > 50%
  normal  : 其余
  即要求"整片区域的大多数像元"都处在高温，贴近干旱的持续/大范围属性。

输入：data\raw\03_thermal_infrared\lst\*.hdf（MOD11A2, 8天合成）
输出：datasets\03_thermal_infrared\drought_tir\images\*.png + json (+ val)
"""
import os
import re
import glob
import json
import numpy as np
from pyhdf.SD import SD, SDC
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 双产品方针（2026-10-15）:
#   本征场 = LST 温度场（Day/Night, °C）—— 已有
#   特色场 = LST 距平场（anomaly）      —— ✱ 近似版，待改
# TODO(待改): 当前"距平"用当景中位数为基准（近似）。真正的 LST 距平需
#             多年同期气候态基准（climatology）；数据齐后替换。

RAW = r"C:\vlmdisaster\data\raw\03_thermal_infrared\lst"
OUT = r"C:\vlmdisaster\datasets\03_thermal_infrared\drought_tir"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_drought_tir.json")

WIN = 128
STRIDE = 192
OUT_SIZE = 384
SCALE = 0.02
DAY_HOT = 35.0      # 白天高温阈值（°C）
NIGHT_HOT = 20.0    # 夜间高温阈值（°C）
FRAC_MIN = 0.50     # 高温像元占比阈值
MAX_PER_FILE = 24
VAL_PER_CLASS = 30
MIN_VALID = 0.30    # 有效像元下限


def load_lst(f):
    h = SD(f, SDC.READ)
    day = h.select("LST_Day_1km")[:].astype("float64")
    ngt = h.select("LST_Night_1km")[:].astype("float64")
    qd = h.select("QC_Day")[:]; qn = h.select("QC_Night")[:]
    h.end()
    day[(day < 7500) | (day > 65535)] = np.nan
    ngt[(ngt < 7500) | (ngt > 65535)] = np.nan
    day[qd > 2] = np.nan; ngt[qn > 2] = np.nan
    return day * SCALE - 273.15, ngt * SCALE - 273.15


def grade(day, ngt):
    """按区域主体判干旱：分位数 + 高温像元占比。"""
    d = day[np.isfinite(day)]
    n = ngt[np.isfinite(ngt)]
    if d.size == 0:
        return None, None, None
    d_p90 = float(np.percentile(d, 90))
    n_p50 = float(np.percentile(n, 50)) if n.size else np.nan
    frac_hot = float(np.mean(d > DAY_HOT))
    if d_p90 > DAY_HOT and (np.isfinite(n_p50) and n_p50 > NIGHT_HOT) and frac_hot > FRAC_MIN:
        return "drought", d_p90, n_p50
    return "normal", d_p90, n_p50


def render(day, ngt, path):
    """本征场：Day/Night LST 温度场"""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=96)
    for ax, a, nm in [(axs[0], day, "Day"), (axs[1], ngt, "Night")]:
        im = ax.imshow(a, cmap="magma", vmin=-20, vmax=60)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(nm + " LST", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("°C", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def render_anom(day, path):
    """特色场：LST 距平场（近似：相对当景中位数）✱待改"""
    base = np.nanmedian(day)
    anom = day - base
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=96)
    im = ax.imshow(anom, cmap="RdBu_r", vmin=-15, vmax=15)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("LST Anomaly (approx)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("ΔT (°C)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    # 清理旧产物，避免 v1 残留污染
    for old in glob.glob(os.path.join(IMG_DIR, "drtir_*.png")):
        os.remove(old)
    files = sorted(glob.glob(os.path.join(RAW, "*.hdf")))
    print(f"文件: {len(files)}")
    records = []
    for f in files:
        m = re.search(r"A(\d{4})(\d{3})\.(h\d{2}v\d{2})", os.path.basename(f))
        if not m:
            continue
        stamp = m.group(1)[-3:] + m.group(2) + "_" + m.group(3)
        try:
            day, ngt = load_lst(f)
        except Exception:
            continue
        H, W = day.shape
        buckets = {}
        for y in range(0, H - WIN + 1, STRIDE):
            for x in range(0, W - WIN + 1, STRIDE):
                dsub = day[y:y + WIN, x:x + WIN]; nsub = ngt[y:y + WIN, x:x + WIN]
                if np.isfinite(dsub).sum() < WIN * WIN * MIN_VALID:
                    continue
                lab, d_p90, n_p50 = grade(dsub, nsub)
                if lab is None:
                    continue
                buckets.setdefault(lab, []).append((d_p90, n_p50, y, x))
        picked = []
        for lab, lst in buckets.items():
            lst.sort(key=lambda t: -t[0])
            picked += [(lab, t) for t in lst[:MAX_PER_FILE]]
        for lab, (d_p90, n_p50, y, x) in picked:
            fn = f"drtir_{stamp}_{y}_{x}.png"
            render(day[y:y + WIN, x:x + WIN], ngt[y:y + WIN, x:x + WIN], os.path.join(IMG_DIR, fn))
            # 特色场：LST 距平（近似）
            fn2 = f"drtir_{stamp}_{y}_{x}_anom.png"
            render_anom(day[y:y + WIN, x:x + WIN], os.path.join(IMG_DIR, fn2))
            records.append({
                "image": fn, "image_feature": fn2, "stamp": stamp, "label": lab,
                "products": ["intrinsic_lst", "feature_anomaly_approx"],
                "day_p90": round(d_p90, 1),
                "night_p50": round(n_p50, 1) if np.isfinite(n_p50) else None,
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a MODIS land-surface-temperature map (day and night) over a region. "
                              "Is this region experiencing drought conditions? Answer drought or normal."},
                    {"from": "gpt",
                     "value": "Yes, this region shows drought conditions (persistently hot day and night)."
                     if lab == "drought" else
                     "No, this region does not show drought conditions."},
                ],
            })
    from collections import Counter, defaultdict
    import random as rnd
    rnd.seed(7)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        rnd.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 4))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)}")


if __name__ == "__main__":
    main()
