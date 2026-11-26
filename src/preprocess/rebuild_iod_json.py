# -*- coding: utf-8 -*-
"""
从已渲的 iod 图片重建 train_iod.json（不重渲图）

对 images/ 里每张 YYYYMMDD.png，按其日期重算 DMI 标签，写回 json。
复用 iod_render.py 的计算函数，保证标签与渲染时口径一致。
"""
import os, re, sys, json
from datetime import date, timedelta

sys.path.insert(0, r"C:\vlmdisaster\src\preprocess")
import iod_render as R

IMG_DIR = os.path.join(R.OUT_DIR, "images")
OUT_JSON = os.path.join(R.OUT_DIR, "train_iod.json")

# 建立日期->文件 索引
all_files = R.glob.glob(os.path.join(R.SST_DIR, "oisst-avhrr-v02r01.*.nc"))
by_date = {}
for f in all_files:
    m = re.search(r"(\d{8})", os.path.basename(f))
    if not m: continue
    ymd = m.group(1)
    if ymd not in by_date or "_preliminary" not in os.path.basename(f):
        by_date[ymd] = f

imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".png"))
print(f"已渲图片: {len(imgs)} 张")

half = 31 // 2
records = []
from collections import Counter
dist = Counter()
skip = 0
for i, fn in enumerate(imgs):
    ymd = fn[:-4]
    if not re.match(r"\d{8}", ymd):
        skip += 1; continue
    d = date.fromisoformat(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
    win = [(d + timedelta(days=k)).strftime("%Y%m%d") for k in range(-half, half + 1)]
    win = [w for w in win if w in by_date]
    if len(win) < 15:
        skip += 1; continue
    fps = [by_date[w] for w in win]
    _, lat0, lon0 = R.load_anom_masked(fps[0])
    mean_field, coords = R.month_mean_anom(fps, lat0, lon0)
    if mean_field is None:
        skip += 1; continue
    dmi = R.dmi_from_field(mean_field, coords[0], coords[1])
    phase = R.iod_phase(dmi)
    if phase is None:
        skip += 1; continue
    png = os.path.join(IMG_DIR, fn)
    answer = f"The Indian Ocean SST anomaly indicates a {phase} phase (DMI={dmi:+.2f})."
    records.append({
        "conversations": [
            {"from": "human", "value": f"<image>{R.PROMPT}"},
            {"from": "gpt", "value": answer},
        ],
        "images": [png],
    })
    dist[phase] += 1
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(imgs)} ...", flush=True)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"\n写成 json: {len(records)} 条 (跳过 {skip}) → {OUT_JSON}")
print(f"标签分布: {dict(dist)}")
