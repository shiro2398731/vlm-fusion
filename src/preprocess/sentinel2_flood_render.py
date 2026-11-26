# -*- coding: utf-8 -*-
r"""
渲染 Sentinel-2 洪涝样本（高质量版：用 NDWI 光谱指数判水体）。

NDWI = (B03 - B08) / (B03 + B08)   （绿光 - 近红外）
  水体 NDWI 高（> 0.0~0.2），植被/裸土 NDWI 低（< 0）。
这是遥感判水体的标准指数，比"颜色阈值"可靠得多。

两类样本（单图 384x384）：
  1) 灾前预测：灾前图 → will_be_flooded / stays_safe
  2) 灾后识别：灾后图 → flooded / not_flooded

输入：台州 tile (T51RUM) 的 S2 景
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
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_s2_flood.json")

TILE = "T42RVQ"       # 巴基斯坦信德省（2022 超级洪灾）
WIN = 128
STRIDE = 192
OUT_SIZE = 384
SCALE_DIV = 4
NDWI_WATER = 0.2     # 窗内 NDWI>此值的算"水"（提高门槛，排除湿土/湿地）
FLOOD_DELTA = 0.15   # 灾后水占比 − 灾前水占比 > 此值 → 判"新增淹没"
VAL_PER_CLASS = 30


def load_scene(zp, div=SCALE_DIV):
    """返回 (rgb, ndwi)，都已降采样。"""
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    def grab(suffix):
        n = [x for x in names if x.endswith(suffix)][0]
        im = Image.open(io.BytesIO(z.read(n)))
        return np.asarray(im.resize((im.width // div, im.height // div), Image.BILINEAR)).astype("float64")
    b03 = grab("B03.jp2")   # 绿
    b08 = grab("B08.jp2")   # 近红外
    b04 = grab("B04.jp2")   # 红（做 RGB）
    b02 = grab("B02.jp2")   # 蓝
    z.close()
    denom = b03 + b08
    denom[denom == 0] = 1
    ndwi = (b03 - b08) / denom
    rgb = np.stack([b04, b03, b02], axis=2)   # 假彩近似（城市用）
    return rgb, ndwi


def water_frac(ndwi_sub):
    return float((ndwi_sub > NDWI_WATER).mean())


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(RAW, f"*{TILE}*.SAFE")))
    items = []
    for zp in zips:
        m = re.search(r"MSIL1C_(\d{8})", os.path.basename(zp))
        items.append((m.group(1), zp))
    items.sort()
    print("台州景:", [d for d, _ in items])
    if len(items) < 2:
        print("景不足"); return
    pre_items = items[:2]
    post_items = items[2:]
    print(f"灾前 {len(pre_items)} 景, 灾后 {len(post_items)} 景")

    print("载入所有景...")
    scenes = []
    for date, zp in pre_items:
        scenes.append(("pre", date, load_scene(zp)))
    for date, zp in post_items:
        try:
            scenes.append(("post", date, load_scene(zp)))
        except Exception as e:
            print(f"  ✗ {date}: {e}")
    base_rgb, base_ndwi = scenes[0][2]
    H, W = base_rgb.shape[:2]
    ys = list(range(0, H - WIN + 1, STRIDE))
    xs = list(range(0, W - WIN + 1, STRIDE))
    print(f"网格 {len(ys)}x{len(xs)} = {len(ys)*len(xs)} 位")

    records = []
    pre_scene = scenes[0]
    post_scenes = [s for s in scenes if s[0] == "post"]
    # 先算每个位置的"灾前水占比"（基准）
    pre_ndwi = pre_scene[2][1]
    pre_wf = {}
    for (y, x) in [(a, b) for a in ys for b in xs]:
        pre_wf[(y, x)] = water_frac(pre_ndwi[y:y + WIN, x:x + WIN])
    for (y, x) in [(a, b) for a in ys for b in xs]:
        ever = False
        later = []
        for _, date, (rgb, ndwi) in post_scenes:
            if y + WIN > rgb.shape[0] or x + WIN > rgb.shape[1]:
                continue
            nw = ndwi[y:y + WIN, x:x + WIN]
            wf = water_frac(nw)
            delta = wf - pre_wf[(y, x)]
            if delta > FLOOD_DELTA:          # 相对增量判淹没
                ever = True
            later.append((date, rgb[y:y + WIN, x:x + WIN], delta))
        # 灾前样本
        b = pre_scene[2][0][y:y + WIN, x:x + WIN]
        lab_pre = "will_be_flooded" if ever else "stays_safe"
        fn_pre = f"s2flood_pre_{pre_scene[1]}_{y}_{x}.png"
        Image.fromarray(b.astype("uint8")).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(os.path.join(IMG_DIR, fn_pre))
        records.append({
            "image": fn_pre, "phase": "pre", "label": lab_pre, "date": pre_scene[1],
            "conversations": [
                {"from": "human",
                 "value": "<image>This is a pre-event Sentinel-2 image of a region. Will this area be flooded soon?"},
                {"from": "gpt",
                 "value": "Yes, this area will be flooded." if ever else "No, this area will stay safe."},
            ],
        })
        # 灾后样本
        for date, rgb_sub, delta in later:
            lab_post = "flooded" if delta > FLOOD_DELTA else "not_flooded"
            fn_post = f"s2flood_post_{date}_{y}_{x}.png"
            Image.fromarray(rgb_sub.astype("uint8")).resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(os.path.join(IMG_DIR, fn_post))
            records.append({
                "image": fn_post, "phase": "post", "label": lab_post, "date": date,
                "water_delta": round(delta, 3),
                "conversations": [
                    {"from": "human",
                     "value": "<image>This is a post-event Sentinel-2 image of a region. Is this area flooded?"},
                    {"from": "gpt",
                     "value": "Yes, this area is flooded." if delta > FLOOD_DELTA else "No, this area is not flooded."},
                ],
            })

    from collections import Counter, defaultdict
    import random as rnd
    rnd.seed(19)
    print(f"\n合计 {len(records)}")
    print("总体标签:", dict(Counter(r["label"] for r in records)))
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
