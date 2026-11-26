# -*- coding: utf-8 -*-
r"""
渲染 TROPOMI SO2 火山样本（滑窗切窗版，修正 swath 结构）。

TROPOMI swath 结构：(4172 行, 450 列)
  - 行 = 沿轨（纬度方向，从极到极）
  - 列 = 跨轨（扫描宽度）
  - 一块 (行区间 x 列区间) ≈ 一小块地理区域
切法：
  - 正样本：块内 SO2 有显著高值（火山羽流）
  - 负样本：块内 SO2 接近背景
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

RAW = r"C:\vlmdisaster\data\raw\11_atmospheric\tropomi\so2"
OUT = r"C:\vlmdisaster\datasets\11_atmospheric\volcano_so2"
IMG_DIR = os.path.join(OUT, "images_fine")
OUT_JSON = os.path.join(OUT, "train_so2_fine.json")

QA_MIN = 0.5
BLK_ROW = 400      # 块高（行）
BLK_COL = 150      # 块宽（列）
STEP_ROW = 200
STEP_COL = 75
OUT_SIZE = 384
POS_MEAN = 0.003
POS_MAX = 0.02
NEG_MEAN = 0.0005
MAX_POS = 6
MAX_NEG = 6
VAL_PER_CLASS = 30


def load(nc):
    with h5py.File(nc, "r") as h:
        P = h["PRODUCT"]
        so2 = P["sulfurdioxide_total_vertical_column"][0].astype("float64")
        lat = P["latitude"][0].astype("float64")
        lon = P["longitude"][0].astype("float64")
        qa = P["qa_value"][0]
    so2[(qa <= QA_MIN) | (so2 < -1)] = np.nan
    return so2, lat, lon


def render(sub, path):
    vmax = 0.01
    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    im = ax.imshow(sub * 1000, cmap="inferno", vmin=0, vmax=vmax * 1000, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("SO2 column", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("mmol/m2", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.nc")))
    print(f"文件: {len(files)}")
    records = []
    for f in files:
        m = re.search(r"(\d{8}T\d{6})", os.path.basename(f))
        stamp = m.group(1) if m else "x"
        try:
            so2, lat, lon = load(f)
        except Exception as e:
            print(f"  ✗ {stamp}: {e}"); continue
        H, W = so2.shape
        pos, neg = [], []
        for y in range(0, H - BLK_ROW + 1, STEP_ROW):
            for x in range(0, W - BLK_COL + 1, STEP_COL):
                sub = so2[y:y + BLK_ROW, x:x + BLK_COL]
                v = sub[np.isfinite(sub)]
                if v.size < 200:
                    continue
                mean = float(np.mean(v)); mx = float(np.max(v))
                info = (mean, mx, y, x)
                if mean > POS_MEAN or mx > POS_MAX:
                    pos.append(info)
                elif mean < NEG_MEAN:
                    neg.append(info)
        pos.sort(key=lambda t: -t[1]); neg.sort(key=lambda t: t[0])
        picked = [("volcanic_so2", t) for t in pos[:MAX_POS]] + \
                 [("no_so2", t) for t in neg[:MAX_NEG]]
        for lab, (mean, mx, y, x) in picked:
            fn = f"so2_{stamp}_{y}_{x}.png"
            render(so2[y:y + BLK_ROW, x:x + BLK_COL], os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "stamp": stamp, "label": lab,
                "mean": round(mean, 5), "max": round(mx, 5),
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a TROPOMI satellite SO2 column map over a region. "
                              "Does this region show volcanic SO2 emission (a plume)?"},
                    {"from": "gpt",
                     "value": "Yes, this region shows significant SO2, likely volcanic emission."
                     if lab == "volcanic_so2" else
                     "No, this region shows no significant SO2 plume."},
                ],
            })
        print(f"  {stamp}: pos {len(pos)} neg {len(neg)} → 取 {len(picked)}")

    from collections import Counter, defaultdict
    import random as _rnd
    _rnd.seed(1)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        _rnd.shuffle(lst)
        k = min(VAL_PER_CLASS, len(lst) // 4)
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} → {OUT_JSON}")
    print(f"val {len(val)} (val 分布 {dict(Counter(r['label'] for r in val))})")


if __name__ == "__main__":
    main()
