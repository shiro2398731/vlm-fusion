# -*- coding: utf-8 -*-
r"""
渲染 Sentinel-2 滑坡样本（高质量版：NDVI 双时相）。

NDVI = (B08 - B04) / (B08 + B04)   （近红外 - 红）
  - 植被 NDVI 高（>0.3）；裸土/滑坡体 NDVI 低。

滑坡在光学上的证据：同一位置"植被 → 裸土"（NDVI 骤降）。
所以用双时相：以最早景为基准，后续景与之对比：
  - landslide_scar : 窗内 NDVI 显著下降（植被被冲毁，疑似滑坡）
  - revegetated    : NDVI 上升
  - stable         : 无显著变化

输入：雅安 tile (T48RTU) 的 S2 景
输出：datasets\01_optical_vnir\landslide\images\*.png + json (+ val)
"""
import os
import re
import io
import sys
import glob
import json
import zipfile
import warnings
import numpy as np
from PIL import Image

warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\landslide"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_landslide.json")

TILE = "T18QXF"       # 海地 Nippes（2021-08-14 7.2级地震同震滑坡，加勒比云少）
WIN = 128
STRIDE = 192
OUT_SIZE = 384
SCALE_DIV = 4
NDVI_DROP = 0.15     # NDVI 下降超过此值 → 疑似滑坡
NDVI_RISE = 0.15
VAL_PER_CLASS = 30


def load_scene(zp, div=SCALE_DIV):
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    def grab(suffix):
        n = [x for x in names if x.endswith(suffix)][0]
        im = Image.open(io.BytesIO(z.read(n)))
        return np.asarray(im.resize((im.width // div, im.height // div), Image.BILINEAR)).astype("float64")
    b08 = grab("B08.jp2"); b04 = grab("B04.jp2")
    b03 = grab("B03.jp2"); b02 = grab("B02.jp2")
    z.close()
    denom = b08 + b04
    denom[denom == 0] = 1
    ndvi = (b08 - b04) / denom
    rgb = np.stack([b04, b03, b02], axis=2)
    return rgb, ndvi


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(RAW, f"*{TILE}*.SAFE")))
    items = sorted((re.search(r"MSIL1C_(\d{8})", os.path.basename(z)).group(1), z) for z in zips)
    # 只取泸定事件年份（2022）的景，排除同 tile 下其他年份（如雅安 2026）
    items = [(d, z) for d, z in items if d.startswith("2021")]
    print("海地景（2021）:", [d for d, _ in items])
    if len(items) < 2:
        print("景不足（需≥2）"); return
    base_date, base_zip = items[0]
    base_rgb, base_ndvi = load_scene(base_zip)
    H, W = base_rgb.shape[:2]
    ys = list(range(0, H - WIN + 1, STRIDE))
    xs = list(range(0, W - WIN + 1, STRIDE))
    print(f"基准景 {base_date}, 网格 {len(ys)}x{len(xs)}")

    records = []
    for date, zp in items[1:]:
        try:
            rgb, ndvi = load_scene(zp)
        except Exception as e:
            print(f"  ✗ {date}: {e}"); continue
        for (y, x) in [(a, b) for a in ys for b in xs]:
            if y + WIN > rgb.shape[0] or x + WIN > rgb.shape[1]:
                continue
            b_nd = base_ndvi[y:y + WIN, x:x + WIN]
            a_nd = ndvi[y:y + WIN, x:x + WIN]
            # 有效植被像素上的平均 NDVI
            bv = b_nd[np.isfinite(b_nd)]; av = a_nd[np.isfinite(a_nd)]
            if bv.size < 100 or av.size < 100:
                continue
            d = float(np.mean(av) - np.mean(bv))
            if d < -NDVI_DROP:
                lab = "landslide_scar"
            elif d > NDVI_RISE:
                lab = "revegetated"
            else:
                lab = "stable"
            fn = f"ls_{base_date}_{date}_{y}_{x}.png"
            Image.fromarray(rgb[y:y + WIN, x:x + WIN].astype("uint8")).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "pre": base_date, "post": date, "label": lab,
                "ndvi_delta": round(d, 3),
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a Sentinel-2 image of a mountainous region, compared with an earlier date. "
                              "Classify the change: landslide_scar (vegetation lost, possible landslide), revegetated, or stable."},
                    {"from": "gpt",
                     "value": {
                         "landslide_scar": "This slope lost vegetation, likely a landslide scar.",
                         "revegetated": "This slope gained vegetation.",
                         "stable": "This slope looks unchanged.",
                     }[lab]},
                ],
            })
        print(f"  {date}: 处理完成")

    from collections import Counter, defaultdict
    import random as rnd
    rnd.seed(23)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        rnd.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 5))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)}")


if __name__ == "__main__":
    main()
