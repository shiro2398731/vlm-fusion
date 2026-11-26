# -*- coding: utf-8 -*-
r"""
渲染 Sentinel-2 光学真彩样本（服务 光学VNIR 源的多个元素：灾后评估类）。

数据：S2 L1C SAFE（zip），用官方合成好的 TCI.jp2（True Color Image，10980x10980 RGB）。
逻辑：
  读 TCI → 切窗（如 512x512）→ 出图。这些"地表影像"作为"灾后/灾前"的底图样本。
  （本批 2 景为白海豚 AOI 附近，可作"洪水/灾后"光学底图）
输出：datasets\01_optical_vnir\<element>\images\*.png + json
"""

import os
import re
import sys
import glob
import json
import io
import zipfile
import warnings
import numpy as np
from PIL import Image

warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\flood_optical"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_s2.json")

WIN = 512
STRIDE = 384
OUT_SIZE = 384
MAX_PER_SCENE = 60       # 每景最多切几个窗（10980/384≈28 → 28*28≈784，太多，抽 60）
VAL_PER_CLASS = 30


def dark_frac(arr):
    """窗口内"很暗"像素占比（水/阴影/裸地）——用于给样本打粗略标签。"""
    g = arr.mean(axis=2)
    return float((g < 40).mean())


def bright_frac(arr):
    g = arr.mean(axis=2)
    return float((g > 200).mean())


def grade(arr):
    """粗略光学地表分级：water(暗多) / urban_bright(亮多) / land(其余)"""
    d = dark_frac(arr); b = bright_frac(arr)
    if d > 0.4:
        return "water_or_dark"
    if b > 0.3:
        return "bright_surface"
    return "land"


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(RAW, "*.SAFE")))
    print(f"SAFE 数: {len(zips)}")
    records = []
    for zp in zips:
        stamp = re.search(r"MSIL1C_(\d{8})", os.path.basename(zp))
        stamp = stamp.group(1) if stamp else "x"
        try:
            z = zipfile.ZipFile(zp)
            tci = [n for n in z.namelist() if n.endswith("TCI.jp2")][0]
            data = z.read(tci)
            z.close()
            im = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            print(f"  ✗ {stamp}: {e}"); continue
        W, H = im.size
        arr = np.asarray(im)
        print(f"  {stamp}: TCI {W}x{H}")
        # 切窗：网格打散，抽取 MAX_PER_SCENE 个
        ys = list(range(0, H - WIN + 1, STRIDE))
        xs = list(range(0, W - WIN + 1, STRIDE))
        import random as rnd
        rnd.seed(1)
        cells = [(y, x) for y in ys for x in xs]
        rnd.shuffle(cells)
        n = 0
        for (y, x) in cells:
            if n >= MAX_PER_SCENE:
                break
            sub = arr[y:y + WIN, x:x + WIN]
            if sub.shape != (WIN, WIN, 3):
                continue
            lab = grade(sub)
            fn = f"s2_{stamp}_{y}_{x}.png"
            Image.fromarray(sub).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "date": stamp, "label": lab,
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a Sentinel-2 true-color satellite image of a region. "
                              "Describe the land-cover type: water_or_dark, bright_surface, or land."},
                    {"from": "gpt",
                     "value": {
                         "water_or_dark": "This region is dominated by dark surfaces such as water or shadow.",
                         "bright_surface": "This region is dominated by bright surfaces such as bare ground or built-up area.",
                         "land": "This region shows mixed land cover.",
                     }[lab]},
                ],
            })
            n += 1
        print(f"      → {n} 窗")

    from collections import Counter, defaultdict
    import random as rnd2
    rnd2.seed(2)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        rnd2.shuffle(lst)
        k = min(VAL_PER_CLASS, len(lst) // 5)
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)} {dict(Counter(r['label'] for r in val))}")


if __name__ == "__main__":
    main()
