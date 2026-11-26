# -*- coding: utf-8 -*-
r"""
CME 日冕物质抛射（源 12 空间天气）—— 渲染

判据（CDAW SOHO/LASCO CME Catalog 自带属性，国际通用）：
  halo（CPA=Halo 或 角宽 >= 360°）  → halo_cme
  速度 >= 1000 km/s                 → fast_cme
  其余                               → normal_cme

双产品：
  本征场 = LASCO C3 白光日冕图像原图（该源的本征观测）
  特色场 = 带 CME 方位标注（CPA 角指示线）

数据：data\raw\12_space_weather\cme\images\*.jpg
     data\raw\12_space_weather\cme\catalog\cme_2024.json

用法: python cme_render.py
"""
import os
import re
import sys
import json
import math
import argparse
import warnings
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image, ImageDraw

RAW = r"C:\vlmdisaster\data\raw\12_space_weather\cme"
OUT = r"C:\vlmdisaster\datasets\12_space_weather\cme"
IMG = os.path.join(OUT, "images")
CAT = os.path.join(RAW, "catalog", "cme_2024.json")
OUT_SIZE = 384
VAL_PER_CLASS = 30

PROMPT = ("This is a SOHO LASCO C3 white-light coronagraph image of the Sun's corona. "
          "Classify the coronal mass ejection: normal_cme, fast_cme, or halo_cme.")

ANS = {"halo_cme": "Halo CME: the ejection spans a full 360 degrees around the Sun.",
       "fast_cme": "Fast CME: ejection speed at or above 1000 km/s.",
       "normal_cme": "Ordinary CME: a moderate-speed, non-halo coronal mass ejection."}


def grade(ev):
    if ev.get("halo"):
        return "halo_cme"
    if ev.get("speed", 0) >= 1000:
        return "fast_cme"
    return "normal_cme"


def render(src, dst, ev, with_mark):
    im = Image.open(src).convert("RGB")
    im = im.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
    if with_mark and ev:
        d = ImageDraw.Draw(im)
        cpa = ev.get("cpa", "")
        cx = cy = OUT_SIZE / 2
        # CPA：从北起顺时针的角度
        try:
            ang = float(str(cpa).replace("Halo", "0"))
        except Exception:
            ang = 0.0
        rad = math.radians(90 - ang)
        r2 = OUT_SIZE * 0.42
        x2 = cx + r2 * math.cos(rad)
        y2 = cy - r2 * math.sin(rad)
        d.line([cx, cy, x2, y2], fill=(255, 90, 40), width=2)
        if ev.get("halo"):
            d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=(255, 220, 60), width=2)
    im.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    for f in os.listdir(IMG):
        if f.endswith(".png"):
            os.remove(os.path.join(IMG, f))

    evs = json.load(open(CAT, encoding="utf-8"))
    files = {f: os.path.join(RAW, "images", f) for f in os.listdir(os.path.join(RAW, "images"))
             if f.endswith(".jpg")}
    print("事件 %d | 图像 %d" % (len(evs), len(files)))

    by = {}
    for e in evs:
        d = e["date"]
        fn = "%s_%02d%02d_c3.jpg" % (d, e["hour"], e["minute"])
        if fn not in files:
            continue
        lab = grade(e)
        by.setdefault(lab, []).append((e, files[fn]))

    print("可渲染:", {k: len(v) for k, v in by.items()})

    import random
    random.seed(21)
    recs, cnt = [], {}
    for lab, lst in by.items():
        random.shuffle(lst)
        for e, p in lst[:args.per_class]:
            cnt[lab] = cnt.get(lab, 0) + 1
            base = "%s_%02d%02d_%s" % (e["date"], e["hour"], e["minute"], lab)
            for sfx, mk, prod in (("_c3", False, "intrinsic_lasco_c3"),
                                  ("_cpa", True, "feature_cme_position_angle")):
                png = os.path.join(IMG, base + sfx + ".png")
                render(p, png, e, mk)
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ANS[lab]}],
                    "images": [png], "product": prod, "label": lab,
                    "speed_kms": e.get("speed"), "cpa": e.get("cpa"),
                    "width": e.get("width"), "stamp": "%s %02d:%02d" % (e["date"], e["hour"], e["minute"])})

    import collections
    by2 = collections.defaultdict(list)
    for r in recs:
        by2[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by2.items():
        random.shuffle(lst)
        n = len(lst)
        k = max(1, min(2, n // 3)) if n <= 8 else min(VAL_PER_CLASS, max(3, n // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_cme.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_cme_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n图 %d | train %d | val %d" % (len(recs), len(train), len(val)))
    print("分布:", dict(cnt))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()
