# -*- coding: utf-8 -*-
r"""
渲染 Sentinel-2 "双时相"洪水样本（简化版，服务光学 VNIR 的洪水/洪涝元素）。

思路：
  以灾前景（最早的）为基准，对每个"灾后"景，在同一 tile 的相同像素位置切窗成对。
  比较两窗的"暗像素比例"（水/淹没区），判定：
    - stable   : 灾前灾后差异小
    - flooded  : 灾后暗像素显著增加（疑似洪水/淹没）
    - receded  : 灾后暗像素减少
  每对生成一张"并排对比图"（左灾前、右灾后），作为样本。

这是"简化版"（未做严格辐射/几何配准），用于先跑通链路；严谨版后续升级。
输出：datasets\01_optical_vnir\flood_optical\images\*.png + json (+ val)
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
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\flood_optical"
IMG_DIR = os.path.join(OUT, "images_dt")
OUT_JSON = os.path.join(OUT, "train_s2_dt.json")

WIN = 512
STRIDE = 768
OUT_SIZE = 384
PAIRS_PER_DATE = 40
VAL_PER_CLASS = 20


def load_tci(zp):
    z = zipfile.ZipFile(zp)
    tci = [n for n in z.namelist() if n.endswith("TCI.jp2")][0]
    data = z.read(tci)
    z.close()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def dark_frac(a):
    g = a.mean(axis=2)
    return float((g < 45).mean())


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(RAW, "*.SAFE")))
    items = []
    for zp in zips:
        m = re.search(r"MSIL1C_(\d{8})", os.path.basename(zp))
        items.append((m.group(1), zp))
    items.sort()
    print("景（按日期）:", [d for d, _ in items])
    base_date, base_zip = items[0]
    base = load_tci(base_zip)
    Hb, Wb = base.shape[:2]
    print(f"基准景 {base_date}: {Wb}x{Hb}")

    records = []
    for date, zp in items[1:]:
        try:
            aft = load_tci(zp)
        except Exception as e:
            print(f"  ✗ {date}: {e}"); continue
        Ha, Wa = aft.shape[:2]
        H = min(Hb, Ha); W = min(Wb, Wa)
        ys = list(range(0, H - WIN + 1, STRIDE))
        xs = list(range(0, W - WIN + 1, STRIDE))
        import random as rnd
        rnd.seed(len(date))
        cells = [(y, x) for y in ys for x in xs]
        rnd.shuffle(cells)
        n = 0
        for (y, x) in cells[:PAIRS_PER_DATE]:
            b = base[y:y + WIN, x:x + WIN]
            a = aft[y:y + WIN, x:x + WIN]
            if b.shape[:2] != (WIN, WIN) or a.shape[:2] != (WIN, WIN):
                continue
            db, da = dark_frac(b), dark_frac(a)
            diff = da - db
            if diff > 0.10:
                lab = "flooded"
            elif diff < -0.10:
                lab = "receded"
            else:
                lab = "stable"
            # 并排图
            canvas = np.zeros((OUT_SIZE, OUT_SIZE * 2, 3), dtype="uint8")
            lb = Image.fromarray(b).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
            la = Image.fromarray(a).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
            canvas[:, :OUT_SIZE] = np.asarray(lb)
            canvas[:, OUT_SIZE:] = np.asarray(la)
            fn = f"s2dt_{base_date}_{date}_{y}_{x}.png"
            Image.fromarray(canvas).save(os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "pre": base_date, "post": date, "label": lab,
                "dark_pre": round(db, 3), "dark_post": round(da, 3),
                "conversations": [
                    {"from": "human",
                     "value": "<image>Left: pre-event Sentinel-2 image. Right: post-event image of the same area. "
                              "Classify the change: flooded, receded, or stable."},
                    {"from": "gpt",
                     "value": {
                         "flooded": "The right image shows a significant increase in dark (water-covered) areas, indicating flooding.",
                         "receded": "The right image shows less water than before, indicating water receded.",
                         "stable": "The area looks largely unchanged between the two dates.",
                     }[lab]},
                ],
            })
            n += 1
        print(f"  {date}: {n} 对窗")

    from collections import Counter, defaultdict
    import random as rnd2
    rnd2.seed(31)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        rnd2.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 5))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)} {dict(Counter(r['label'] for r in val))}")


if __name__ == "__main__":
    main()
