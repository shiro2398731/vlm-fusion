# -*- coding: utf-8 -*-
r"""
渲染 MOD08 沙尘暴样本（服务"沙尘暴"元素，源 11 大气成分）。

与雾霾的区别：沙尘主要发生在**干旱/沙漠带**（北非撒哈拉、阿拉伯半岛、中亚、中国西北等）。
所以用"Deep Blue 陆地 AOD"+ 纬度带限定（|lat| 在 15~45 且 AOD 高）来近似"沙尘"。

数据：MOD08_D3 HDF4，Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean（180x360，scale 0.001）
输出：datasets\11_atmospheric\sand_dust\images\*.png + json (+ val)
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
OUT = r"C:\vlmdisaster\datasets\11_atmospheric\sand_dust"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_sanddust.json")

VAR = "Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean"
SCALE = 0.001
WIN = 30
STRIDE = 15
OUT_SIZE = 384
# 沙尘带：纬度 15~45°N（撒哈拉/阿拉伯/中亚/西北）为主
LAT_MIN, LAT_MAX = 10, 50
DUST_SEVERE = 1.4     # 窗内峰值 > 此值 视为沙尘暴
DUST_MOD = 0.22       # 窗内均值 > 此值 视为中等沙尘
VAL_PER_CLASS = 30
MAX_PER_CLASS = 24


def load(f):
    h = SD(f, SDC.READ)
    a = h.select(VAR)[:].astype("float64") * SCALE
    h.end()
    a[(a <= 0) | (a > 5) | ~np.isfinite(a)] = np.nan
    return a


def grade(mean, mx):
    if mx > DUST_SEVERE:
        return "dust_storm"
    if mean > DUST_MOD:
        return "dust"
    return "clear"


def render(sub, path):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    im = ax.imshow(sub, cmap="YlOrBr", vmin=0, vmax=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Dust AOD (land)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("AOD", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.hdf")))
    print(f"文件: {len(files)}")
    # 纬度从 90~-90 映射到 180 行（第0行=+90）
    lat_of_row = lambda y: 90 - (y + 0.5)
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
            latc = lat_of_row(y)
            if not (LAT_MIN <= latc <= LAT_MAX):
                continue
            for x in range(0, W - WIN + 1, STRIDE):
                sub = aod[y:y + WIN, x:x + WIN]
                v = sub[np.isfinite(sub)]
                if v.size < WIN * WIN * 0.3:
                    continue
                mean = float(np.mean(v)); mx = float(np.max(v))
                buckets.setdefault(grade(mean, mx), []).append((mean, y, x))
        picked = []
        for lab, lst in buckets.items():
            lst.sort(key=lambda t: t[0])
            if lab == "dust_storm":
                picked += [(lab, t) for t in lst[-MAX_PER_CLASS:]]
            else:
                picked += [(lab, t) for t in lst[:MAX_PER_CLASS]]
        for lab, (mean, y, x) in picked:
            fn = f"dust_{stamp}_{y}_{x}.png"
            render(aod[y:y + WIN, x:x + WIN], os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "date": stamp, "label": lab,
                "mean_aod": round(mean, 3),
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a MODIS land aerosol-optical-depth map (desert/dust belt). "
                              "Classify the sand-dust level: clear, dust, or dust_storm."},
                    {"from": "gpt",
                     "value": {
                         "clear": "This region is clear, no significant dust.",
                         "dust": "This region has moderate dust aerosol.",
                         "dust_storm": "This region has a severe sand-dust storm.",
                     }[lab]},
                ],
            })
        print(f"  {stamp}: {len(picked)} 窗 ({dict((k,len(v)) for k,v in buckets.items())})")

    from collections import Counter, defaultdict
    import random as _rnd
    _rnd.seed(13)
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
