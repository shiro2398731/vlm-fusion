# -*- coding: utf-8 -*-
r"""
渲染 MOD08 气溶胶光学厚度（AOD）样本（服务"雾_霾 / 沙尘暴"元素，源 11 大气成分）。

数据：MOD08_D3 (HDF4)，AOD_550_Dark_Target_Deep_Blue_Combined_Mean，180x360 全球 1°网格，
      scale_factor=0.001，真实 AOD = 原始值 * 0.001。
逻辑：切窗 → 按窗内平均 AOD 分级：
  - clear   : < 0.2  (晴朗/低气溶胶)
  - hazy    : 0.2 ~ 0.6 (轻雾霾)
  - severe  : > 0.6  (重雾霾/沙尘)
输出：datasets\11_atmospheric\haze\images\*.png + json (+ val)
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

RAW = r"C:\vlmdisaster\data\raw\11_atmospheric\mod08"
OUT = r"C:\vlmdisaster\datasets\11_atmospheric\haze"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_haze.json")

VAR = "AOD_550_Dark_Target_Deep_Blue_Combined_Mean"
SCALE = 0.001
WIN = 30          # 窗口边长（180x360 网格，30 像素 ≈ 30°）
STRIDE = 15
OUT_SIZE = 384
CLEAR = 0.2
SEVERE = 0.6
MAX_PER_CLASS = 24
VAL_PER_CLASS = 30


def load(f):
    h = SD(f, SDC.READ)
    s = h.select(VAR)
    a = s[:].astype("float64") * SCALE
    h.end()
    a[(a <= 0) | (a > 5) | ~np.isfinite(a)] = np.nan
    return a


def grade(m, mx=None):
    # 用"窗内均值"判常规，"窗内最大值"补抓局部重霾
    if mx is not None and mx > 1.5:
        return "severe_haze"
    if m < CLEAR:
        return "clear"
    if m > SEVERE:
        return "severe_haze"
    return "haze"


def render(sub, path):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    im = ax.imshow(sub, cmap="YlOrBr", vmin=0, vmax=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("MODIS AOD", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("AOD", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.hdf")))
    print(f"文件: {len(files)}")
    records = []
    for f in files:
        m = re.search(r"A(\d{7})", os.path.basename(f))
        stamp = m.group(1) if m else "x"
        try:
            aod = load(f)
        except Exception as e:
            print(f"  ✗ {stamp}: {e}"); continue
        H, W = aod.shape
        buckets = {}
        for y in range(0, H - WIN + 1, STRIDE):
            for x in range(0, W - WIN + 1, STRIDE):
                sub = aod[y:y + WIN, x:x + WIN]
                v = sub[np.isfinite(sub)]
                if v.size < WIN * WIN * 0.4:
                    continue
                mean = float(np.mean(v))
                mx = float(np.max(v))
                buckets.setdefault(grade(mean, mx), []).append((mean, y, x))
        picked = []
        for lab, lst in buckets.items():
            lst.sort(key=lambda t: t[0])
            if lab == "severe_haze":
                picked += [(lab, t) for t in lst[-MAX_PER_CLASS:]]
            else:
                picked += [(lab, t) for t in lst[:MAX_PER_CLASS]]
        for lab, (mean, y, x) in picked:
            fn = f"haze_{stamp}_{y}_{x}.png"
            render(aod[y:y + WIN, x:x + WIN], os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "date": stamp, "label": lab,
                "mean_aod": round(mean, 3),
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a MODIS aerosol-optical-depth (AOD) map over a region. "
                              "Classify the atmospheric haze/sand-dust level: clear, haze, or severe_haze."},
                    {"from": "gpt",
                     "value": {
                         "clear": "This region has clear air, low aerosol loading.",
                         "haze": "This region is hazy, with moderate aerosol loading.",
                         "severe_haze": "This region has severe haze or dust, with very high aerosol loading.",
                     }[lab]},
                ],
            })
        print(f"  {stamp}: {len(picked)} 窗 ({dict((k,len(v)) for k,v in buckets.items())})")

    from collections import Counter, defaultdict
    import random as _rnd
    _rnd.seed(5)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        _rnd.shuffle(lst)
        k = min(VAL_PER_CLASS, len(lst) // 5)
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)} {dict(Counter(r['label'] for r in val))}")


if __name__ == "__main__":
    main()
