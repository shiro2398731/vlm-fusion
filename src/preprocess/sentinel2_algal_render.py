# -*- coding: utf-8 -*-
r"""
渲染 Sentinel-2 赤潮/绿潮样本（服务光学 VNIR 的"赤潮_绿潮"元素）。

判据：藻类指数 NDCI = (B05 − B04) / (B05 + B04)（红边-红，对藻类敏感）。
  藻华区域 NDCI 高、且水体反射偏绿（B03 相对 B04 高）。
分级：
  - algal_bloom : 窗内 NDCI 均值高（疑似赤潮/绿潮）
  - normal_water: 普通水体
  - land_or_other: 非水（陆地/云）

输入：东海 tile (T51RVP) 的 S2 景
输出：datasets\01_optical_vnir\algal_bloom\images\*.png + json (+ val)
"""
import os
import re
import io
import glob
import json
import zipfile
import warnings
import numpy as np
from PIL import Image

warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\algal_bloom"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_algal_bloom.json")

TILE = "T51RVP"
WIN = 128
STRIDE = 192
OUT_SIZE = 384
SCALE_DIV = 4
NDCI_BLOOM = 0.02     # 窗内 NDCI 均值 > 此值 → 疑似藻华
MAX_PER_CLASS = 40
VAL_PER_CLASS = 30


def load_scene(zp, div=SCALE_DIV):
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    TARGET = 10980 // div      # 统一目标尺寸（10m 波段降采样后的大小）
    def grab(suffix):
        n = [x for x in names if x.endswith(suffix)][0]
        im = Image.open(io.BytesIO(z.read(n)))
        # 不问原分辨率，一律 resize 到统一尺寸（20m 和 10m 波段对齐）
        im = im.resize((TARGET, TARGET), Image.BILINEAR)
        return np.asarray(im).astype("float64") / 10000.0   # S2 jp2 为 16 位反射率（×10000）
    b04 = grab("B04.jp2"); b05 = grab("B05.jp2")
    b03 = grab("B03.jp2"); b02 = grab("B02.jp2")
    z.close()
    d = b05 + b04; d[d == 0] = 1
    ndci = (b05 - b04) / d
    rgb = np.stack([b04, b03, b02], axis=2)
    return rgb, ndci


def grade(ndci_sub, rgb_sub):
    v = ndci_sub[np.isfinite(ndci_sub)]
    if v.size < 100:
        return None
    m = float(np.mean(v))
    g = rgb_sub.mean(axis=2)
    # 反射率尺度（0~1）：亮（>0.45）占比高 → 云/亮陆（放宽，让近岸浑浊水也能进）
    if (g > 0.45).mean() > 0.7:
        return "land_or_other"
    if m > NDCI_BLOOM:
        return "algal_bloom"
    return "normal_water"


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(RAW, f"*{TILE}*.SAFE")))
    print(f"赤潮景: {len(zips)}")
    records = []
    import random as rnd
    rnd.seed(3)
    for zp in zips:
        date = re.search(r"MSIL1C_(\d{8})", os.path.basename(zp)).group(1)
        try:
            rgb, ndci = load_scene(zp)
        except Exception as e:
            print(f"  ✗ {date}: {e}"); continue
        H, W = rgb.shape[:2]
        ys = list(range(0, H - WIN + 1, STRIDE))
        xs = list(range(0, W - WIN + 1, STRIDE))
        cells = [(y, x) for y in ys for x in xs]
        rnd.shuffle(cells)
        buckets = {}
        for (y, x) in cells:
            lab = grade(ndci[y:y + WIN, x:x + WIN], rgb[y:y + WIN, x:x + WIN])
            if lab:
                buckets.setdefault(lab, []).append((y, x))
        picked = []
        for lab, lst in buckets.items():
            for (y, x) in lst[:MAX_PER_CLASS]:
                picked.append((lab, y, x))
        for lab, y, x in picked:
            fn = f"algae_{date}_{y}_{x}.png"
            Image.fromarray(rgb[y:y + WIN, x:x + WIN].astype("uint8")).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "date": date, "label": lab,
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a Sentinel-2 true-color image of coastal water. "
                              "Classify: algal_bloom (red/green tide), normal_water, or land_or_other."},
                    {"from": "gpt",
                     "value": {
                         "algal_bloom": "This water shows an algal bloom (red or green tide).",
                         "normal_water": "This is normal coastal water.",
                         "land_or_other": "This area is land or cloud, not water.",
                     }[lab]},
                ],
            })
        print(f"  {date}: {len(picked)} 窗 {dict((k,len(v)) for k,v in buckets.items())}")

    from collections import Counter, defaultdict
    rnd2 = __import__("random"); rnd2.seed(4)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        rnd2.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 4))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)}")


if __name__ == "__main__":
    main()
