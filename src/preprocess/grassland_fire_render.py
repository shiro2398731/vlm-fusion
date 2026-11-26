# -*- coding: utf-8 -*-
r"""
FIRMS 火点渲染 → 草原火灾样本（服务热红外 TIR 的"草原火灾"元素）。

⚠️ 方法说明与局限（务必保留）：
  FIRMS 火点 CSV 只有经纬度/亮度/frp，**没有地表覆盖类型字段**，无法从数据本身
  判定"这场火烧的是草原还是森林"。本脚本采用**已知草原/萨瓦纳区域地理框**近似：
  凡落在下列地理框内的显著火点 → 视为草原火灾；否则不取。
  这是一个**近似判据**，会与相邻的农田/灌丛火灾混淆，精度有限，已如实标注。

  区域框（lon_min, lon_max, lat_min, lat_max）：
    1. 欧亚草原带（哈萨克斯坦-蒙古-内蒙古）
    2. 北美大平原
    3. 非洲萨瓦纳（南北两带）
    4. 南美潘帕斯/塞拉多
    5. 澳洲草原

判据：某地理框内、某天显著火点（FRP>=5MW）数 >= MIN_HOT → 该框该天判"草原火灾"，
      否则为"无草原火灾"（真负样本）。同一框多天 → 天然构成正负样本。

输入：data\raw\03_thermal_infrared\firms\history\fire_*.csv
输出：datasets\03_thermal_infrared\grassland_fire\images\*.png + json (+ val)

用法：
  python src/preprocess/grassland_fire_render.py --limit-days 5   # 先小批量
  python src/preprocess/grassland_fire_render.py                  # 全量
"""
import os
import sys
import csv
import glob
import json
import argparse
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

FIRE_DIR = r"C:\vlmdisaster\data\raw\03_thermal_infrared\firms\history"
OUT = r"C:\vlmdisaster\datasets\03_thermal_infrared\grassland_fire"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_grassland_fire.json")

FIG_SIZE = (4.5, 4.5)
DPI = 100
OUT_SIZE = 384
FRP_MIN = 5.0
MIN_HOT = 30         # 框内显著火点数下限（提高→更多 "no" 负样本）
VAL_PER_CLASS = 30
MAX_PER_BOX_DAY = 1   # 每框每天最多 1 张图

# 草原/萨瓦纳区域框： (名称, lon_min, lon_max, lat_min, lat_max)
GRASS_BOXES = [
    ("CentralAsia_Steppe",  45.0, 120.0,  40.0,  55.0),
    ("NorthAmerica_Plains", -105.0, -95.0, 32.0,  49.0),
    ("Africa_Savanna_N",    -15.0,  40.0,   8.0,  16.0),
    ("Africa_Savanna_S",    15.0,  35.0, -25.0, -12.0),
    ("SouthAmerica_Pampas", -65.0, -55.0, -38.0, -30.0),
    ("Australia_Grassland", 130.0, 150.0, -35.0, -20.0),
]

PROMPT = ("This is a satellite fire-hotspot map over one region. "
          "Is there a significant grassland wildfire in this region? Answer Yes or No.")


def load_points(path):
    pts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                pts.append((float(r["latitude"]), float(r["longitude"]),
                            float(r.get("frp", 0) or 0)))
            except (ValueError, KeyError, TypeError):
                continue
    return pts


def render(pts, box, out_png, title):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    _, lon0, lon1, lat0, lat1 = box
    if pts:
        lats = np.array([p[0] for p in pts])
        lons = np.array([p[1] for p in pts])
        frps = np.array([p[2] for p in pts])
        vmax = max(float(min(frps.max(), 200)), 1.0)
        ax.scatter(lons, lats, c=frps, cmap="YlOrRd", s=12,
                   alpha=0.85, vmin=0, vmax=vmax)
    else:
        ax.text(0.5, 0.5, "No fire hotspot", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=11)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    PILImage.open(out_png).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(FIRE_DIR, "fire_*.csv")))
    if args.limit_days:
        files = files[:args.limit_days]
    print(f"火点文件: {len(files)}")

    records = []
    for f in files:
        date = os.path.basename(f).replace("fire_", "").replace(".csv", "")
        if os.path.getsize(f) < 1024:
            continue
        pts = load_points(f)
        if not pts:
            continue
        for box in GRASS_BOXES:
            name, lon0, lon1, lat0, lat1 = box
            inpts = [p for p in pts if lat0 <= p[0] <= lat1 and lon0 <= p[1] <= lon1]
            hot = [p for p in inpts if p[2] >= FRP_MIN]
            lab = "yes" if len(hot) >= MIN_HOT else "no"
            fn = f"glfire_{date}_{name}.png"
            render(inpts, box, os.path.join(IMG_DIR, fn), f"{name} {date}")
            records.append({
                "image": fn, "date": date, "region": name, "label": lab,
                "n_points": len(inpts),
                "n_hot": len(hot),
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt",
                     "value": "Yes, there is a significant grassland wildfire in this region."
                     if lab == "yes" else
                     "No, there is no significant grassland wildfire in this region."},
                ],
            })

    from collections import Counter, defaultdict
    import random as rnd
    rnd.seed(7)
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
    print(f"train {len(train)} | val {len(val)} {dict(Counter(r['label'] for r in val))}")


if __name__ == "__main__":
    main()
