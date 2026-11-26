# -*- coding: utf-8 -*-
r"""
渲染 SMAP 土壤湿度样本（服务"干旱"元素，源 4 被动微波）。

数据：SMAP L3 (HDF5)，406x964 EASE 网格，soil_moisture 单位 m3/m3。
逻辑：切窗 → 按窗内平均土壤湿度分级：
  - dry      : 平均 < 0.1  (偏干)
  - normal   : 0.1 ~ 0.3   (正常)
  - wet      : > 0.3       (偏湿，作对照)
输出：datasets\04_passive_microwave\drought\images\*.png + json (+ val)
"""
import os
import re
import sys
import glob
import json
import numpy as np
import h5py
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = r"C:\vlmdisaster\data\raw\04_passive_microwave\smap"
OUT = r"C:\vlmdisaster\datasets\04_passive_microwave\drought"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_smap.json")

WIN = 24          # 窗口边长（像素，网格 406x964）——小窗才能体现局部干湿
STRIDE = 12
OUT_SIZE = 384
DRY = 0.1
WET = 0.3
MAX_PER_CLASS = 40    # 每档每文件最多取几个
VAL_PER_CLASS = 30


def load(f):
    with h5py.File(f, "r") as h:
        g = h["Soil_Moisture_Retrieval_Data_AM"]
        sm = g["soil_moisture"][:].astype("float64")
    sm[(sm <= 0) | (sm >= 1) | ~np.isfinite(sm)] = np.nan
    return sm


def grade(m):
    if m < DRY:
        return "dry"
    if m > WET:
        return "wet"
    return "normal"


def render(sub, path):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    im = ax.imshow(sub, cmap="BrBG", vmin=0, vmax=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("SMAP soil moisture", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("m3/m3", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.h5")))
    print(f"文件: {len(files)}")
    records = []
    for f in files:
        m = re.search(r"(\d{8})", os.path.basename(f))
        stamp = m.group(1) if m else "x"
        try:
            sm = load(f)
        except Exception as e:
            print(f"  ✗ {stamp}: {e}"); continue
        H, W = sm.shape
        buckets = {}
        for y in range(0, H - WIN + 1, STRIDE):
            for x in range(0, W - WIN + 1, STRIDE):
                sub = sm[y:y + WIN, x:x + WIN]
                v = sub[np.isfinite(sub)]
                if v.size < WIN * WIN * 0.8:   # 有效点太少（海/无效）
                    continue
                mean = float(np.mean(v))
                buckets.setdefault(grade(mean), []).append((mean, y, x))
        picked = []
        for lab, lst in buckets.items():
            lst.sort(key=lambda t: t[0])
            if lab == "normal":
                picked += [(lab, t) for t in lst[:MAX_PER_CLASS]]
            elif lab == "dry":
                # 偏干：取最干的若干个（list 按均值升序）
                picked += [(lab, t) for t in lst[:MAX_PER_CLASS]]
            else:
                # 偏湿：取最湿的若干个
                picked += [(lab, t) for t in lst[-MAX_PER_CLASS:]]
        for lab, (mean, y, x) in picked:
            fn = f"smap_{stamp}_{y}_{x}.png"
            render(sm[y:y + WIN, x:x + WIN], os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "date": stamp, "label": lab,
                "mean_sm": round(mean, 4),
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a SMAP satellite soil-moisture map over a region. "
                              "Classify soil moisture: dry, normal, or wet."},
                    {"from": "gpt",
                     "value": {
                         "dry": "This region has dry soil, low soil moisture.",
                         "normal": "This region has normal soil moisture.",
                         "wet": "This region has wet soil, high soil moisture.",
                     }[lab]},
                ],
            })
        print(f"  {stamp}: {len(picked)} 窗 ({dict((k,len(v)) for k,v in buckets.items())})")

    from collections import Counter, defaultdict
    import random as _rnd
    _rnd.seed(3)
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
