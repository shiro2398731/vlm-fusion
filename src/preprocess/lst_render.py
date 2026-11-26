# -*- coding: utf-8 -*-
r"""
渲染 MODIS LST 温度灾害样本（寒潮/霜冻/热浪/正常）— v2 判据重做版。

v1 问题（已废弃）：判据用窗口内"极端值"（nanmax 白天 / nanmin 夜间），
  窗口里只要存在一个小冷点/小热点，整格就被定类，标签反映的是"有个极值点"，
  不是"这片区域在经历温度灾害"。

v2 判据（本版）：分位数 + 像元占比，让"区域主体"决定标签。
  cold_wave : 夜间 P10 < -25°C 且 低温像元(< -25)占比 > 50%
  frost     : 夜间 P10 < -6°C 且 夜间 P50 < 0°C 且 白天 P50 > 0
  heatwave  : 白天 P90 > 35°C 且 高温像元(> 35)占比 > 50%
  normal    : 其余
  （freeze "白天也<0" 样本罕见，仍并入 cold_wave。）

输入：data\raw\03_thermal_infrared\lst\*.hdf（MOD11A2, 8天合成）
输出：datasets\03_thermal_infrared\lst_temperature_disaster\images\*.png + json (+ val)
"""
import os
import re
import sys
import glob
import json
import numpy as np
from pyhdf.SD import SD, SDC
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 双产品方针（2026-10-15）:
#   本征场 = LST 温度场（Day/Night, °C）—— 已有
#   特色场 = LST 距平场（anomaly）      —— ✱ 近似版，待改
# TODO(待改): 真正的 LST 距平需多年同期气候态基准；现用当景中位数近似。

RAW = r"C:\vlmdisaster\data\raw\03_thermal_infrared\lst"
OUT = r"C:\vlmdisaster\datasets\03_thermal_infrared"
ELEMENT = "lst_temperature_disaster"
IMG_DIR = os.path.join(OUT, ELEMENT, "images")
OUT_JSON = os.path.join(OUT, ELEMENT, "train_lst.json")

WIN = 256
STRIDE = 128
OUT_SIZE = 384
SCALE = 0.02
VAL_PER_CLASS = 30
MAX_PER_FILE = 24
MIN_VALID = 0.30

COLD_NIGHT = -25.0     # 夜间极端低温
FROST_NIGHT = -6.0     # 夜间霜冻阈值
HEAT_DAY = 35.0        # 白天高温阈值
FRAC_MIN = 0.50        # 极端像元占比阈值


def load_lst(f):
    h = SD(f, SDC.READ)
    day = h.select("LST_Day_1km")[:].astype("float64")
    ngt = h.select("LST_Night_1km")[:].astype("float64")
    qc_d = h.select("QC_Day")[:]; qc_n = h.select("QC_Night")[:]
    h.end()
    day[(day < 7500) | (day > 65535)] = np.nan
    ngt[(ngt < 7500) | (ngt > 65535)] = np.nan
    day[qc_d > 2] = np.nan
    ngt[qc_n > 2] = np.nan
    return day * SCALE - 273.15, ngt * SCALE - 273.15


def grade(day, ngt):
    d = day[np.isfinite(day)]
    n = ngt[np.isfinite(ngt)]
    if d.size == 0 or n.size == 0:
        return None, np.nan, np.nan
    d_p90 = float(np.percentile(d, 90))
    d_p50 = float(np.percentile(d, 50))
    n_p10 = float(np.percentile(n, 10))
    n_p50 = float(np.percentile(n, 50))
    frac_cold = float(np.mean(n < COLD_NIGHT))
    frac_frost = float(np.mean(n < FROST_NIGHT))
    frac_hot = float(np.mean(d > HEAT_DAY))
    if n_p10 < COLD_NIGHT and frac_cold > FRAC_MIN:
        return "cold_wave", d_p90, n_p10
    if n_p10 < FROST_NIGHT and n_p50 < 0.0 and d_p50 > 0.0 and frac_frost > FRAC_MIN:
        return "frost", d_p90, n_p10
    if d_p90 > HEAT_DAY and frac_hot > FRAC_MIN:
        return "heatwave", d_p90, n_p10
    return "normal", d_p90, n_p10


def render(day, ngt, path):
    """本征场：Day/Night LST 温度场"""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=96)
    for ax, a, name in [(axs[0], day, "Day"), (axs[1], ngt, "Night")]:
        im = ax.imshow(a, cmap="magma", vmin=-30, vmax=50)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name + " LST", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("°C", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def render_anom(day, path):
    """特色场：LST 距平场（近似：相对当景中位数）✱待改"""
    anom = day - np.nanmedian(day)
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=96)
    im = ax.imshow(anom, cmap="RdBu_r", vmin=-15, vmax=15)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("LST Anomaly (approx)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("ΔT (°C)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(IMG_DIR, "lst_*.png")):
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
        except Exception as e:
            print(f"  x {stamp}: {e}"); continue
        H, W = day.shape
        buckets = {}
        for y in range(0, H - WIN + 1, STRIDE):
            for x in range(0, W - WIN + 1, STRIDE):
                dsub = day[y:y + WIN, x:x + WIN]
                nsub = ngt[y:y + WIN, x:x + WIN]
                if np.isfinite(dsub).sum() < WIN * WIN * MIN_VALID:
                    continue
                lab, d_p90, n_p10 = grade(dsub, nsub)
                if lab is None:
                    continue
                buckets.setdefault(lab, []).append((d_p90, n_p10, y, x))
        picked = []
        for lab, lst in buckets.items():
            lst.sort(key=lambda t: -(t[0] - (t[1] if np.isfinite(t[1]) else t[0])))
            picked += [(lab, t) for t in lst[:MAX_PER_FILE]]
        for lab, (d_p90, n_p10, y, x) in picked:
            fn = f"lst_{stamp}_{y}_{x}.png"
            render(day[y:y + WIN, x:x + WIN], ngt[y:y + WIN, x:x + WIN], os.path.join(IMG_DIR, fn))
            fn2 = f"lst_{stamp}_{y}_{x}_anom.png"
            render_anom(day[y:y + WIN, x:x + WIN], os.path.join(IMG_DIR, fn2))
            records.append({
                "image": fn, "image_feature": fn2, "stamp": stamp, "label": lab,
                "products": ["intrinsic_lst", "feature_anomaly_approx"],
                "day_p90": round(d_p90, 1),
                "night_p10": round(n_p10, 1) if np.isfinite(n_p10) else None,
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a MODIS land-surface-temperature map (day and night) over a region. "
                              "Classify the temperature hazard: cold_wave, frost, heatwave, or normal."},
                    {"from": "gpt",
                     "value": {
                         "cold_wave": "This region is experiencing a cold wave or freezing conditions, with extreme low temperatures.",
                         "frost": "This region is at risk of frost, with night temperatures below freezing.",
                         "heatwave": "This region is experiencing a heatwave, with very high surface temperatures.",
                         "normal": "This region shows normal temperatures, no temperature hazard.",
                     }[lab]},
                ],
            })
        print(f"  {stamp}: {len(picked)} 窗 ({dict((k,len(v)) for k,v in buckets.items())})")

    from collections import Counter, defaultdict
    import random as _rnd
    _rnd.seed(7)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        _rnd.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 5))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} -> {OUT_JSON}")
    print(f"val {len(val)} {dict(Counter(r['label'] for r in val))}")


if __name__ == "__main__":
    main()
